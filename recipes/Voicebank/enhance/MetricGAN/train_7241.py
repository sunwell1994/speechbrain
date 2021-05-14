#!/usr/bin/env/python3
"""
Recipe for training a speech enhancement system with the Voicebank dataset.

To run this recipe, do the following:
> python train.py hparams/{hyperparam_file}.yaml

Authors
 * Szu-Wei Fu 2020
 * Peter Plantinga 2021
"""

import os
import sys
import shutil
import torch
import torchaudio
import speechbrain as sb
import pickle
import time
from pesq import pesq
from enum import Enum, auto
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import MetricStats
from speechbrain.processing.features import spectral_magnitude
from speechbrain.nnet.loss.stoi_loss import stoi_loss
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.sampler import ReproducibleWeightedRandomSampler


def pesq_eval(pred_wav, target_wav):
    """Normalized PESQ (to 0-1)"""
    return (
        pesq(fs=16000, ref=target_wav.numpy(), deg=pred_wav.numpy(), mode="wb")
        + 0.5
    ) / 5



class SubStage(Enum):
    """For keeping track of training stage progress"""

    GENERATOR = auto()
    CURRENT = auto()
    HISTORICAL = auto()


class MetricGanBrain(sb.Brain):
    def compute_feats(self, wavs):
        """Feature computation pipeline"""
        feats = self.hparams.compute_STFT(wavs)
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)
        return feats

    def compute_forward(self, batch, stage):
        "Given an input batch computes the enhanced signal"
        batch = batch.to(self.device)

        if self.sub_stage == SubStage.HISTORICAL:
            predict_wav, lens = batch.enh_sig
        else:
            noisy_wav, lens = batch.noisy_sig
            noisy_spec = self.compute_feats(noisy_wav)

            # mask with "signal approximation (SA)"
            # print("check noisy spec",noisy_spec.size())
            mask = self.modules.generator(noisy_spec, lengths=lens)
            mask = mask.clamp(min=self.hparams.min_mask).squeeze(2)
            predict_spec = torch.mul(mask, noisy_spec)

            # Also return predicted wav
            predict_wav = self.hparams.resynth(
                torch.expm1(predict_spec), noisy_wav
            )

        return predict_wav

    def compute_objectives(self, predictions, batch, stage, optim_name=""):
        "Given the network predictions and targets compute the total loss"
        predict_wav = predictions
        predict_spec = self.compute_feats(predict_wav)

        clean_wav, lens = batch.clean_sig
        clean_spec = self.compute_feats(clean_wav)
        # print(clean_wav.size(), predict_wav.size())
        # print(clean_spec.size(), predict_spec.size())
        mse_cost = self.hparams.compute_cost(predict_spec, clean_spec, lens)

        # print("batch.id", batch.id)
        ids = self.compute_ids(batch.id, optim_name)

        # One is real, zero is fake
        if optim_name == "generator" or optim_name == "":
            if optim_name == "generator":
                target_score = torch.ones(self.batch_size, 1, device=self.device)
            else:
                target_score = torch.ones(1, 1, device=self.device)
            # if optim_name == "":
            #     print("predict_wav", predict_wav.size(), clean_wav.size())
            #     print("predict_spec: ", predict_spec.size(), clean_spec.size())
            est_score = self.est_score(predict_spec, clean_spec)
            self.mse_metric.append(
                ids, predict_spec, clean_spec, lens, reduction="batch"
            )

        # D Learns to estimate the scores of clean speech
        elif optim_name == "D_clean":
            target_score = torch.ones(self.batch_size, 1, device=self.device)
            est_score = self.est_score(clean_spec, clean_spec)

        # D Learns to estimate the scores of enhanced speech
        elif optim_name == "D_enh" and self.sub_stage == SubStage.CURRENT:
            target_score = self.score(ids, predict_wav, clean_wav, lens)
            est_score = self.est_score(predict_spec, clean_spec)

            # Write enhanced wavs during discriminator training, because we
            # compute the actual score here and we can save it
            self.write_wavs(batch.id, ids, predict_wav, target_score, lens)

        # D Relearns to estimate the scores of previous epochs
        elif optim_name == "D_enh" and self.sub_stage == SubStage.HISTORICAL:
            target_score = batch.score.unsqueeze(1).float()
            est_score = self.est_score(predict_spec, clean_spec)

        # D Learns to estimate the scores of noisy speech
        elif optim_name == "D_noisy":
            noisy_wav, _ = batch.noisy_sig
            noisy_spec = self.compute_feats(noisy_wav)
            target_score = self.score(ids, noisy_wav, clean_wav, lens)
            est_score = self.est_score(noisy_spec, clean_spec)

            # Save scores of noisy wavs
            self.save_noisy_scores(ids, target_score)

        else:
            raise ValueError(f"{optim_name} is not a valid 'optim_name'")

        # Compute the cost
        adv_cost = self.hparams.compute_cost(est_score, target_score)
        if optim_name == "generator":
            adv_cost += self.hparams.mse_weight * mse_cost
            self.metrics["G"].append(adv_cost.detach())
        else:
            self.metrics["D"].append(adv_cost.detach())

        # On validation data compute scores
        if stage != sb.Stage.TRAIN:
            # Evaluate speech quality/intelligibility
            if self.hparams.mode == 'val':
                if self.hparams.target_metric == "stoi":
                    self.stoi_metric.append(
                        batch.id, predict_wav, clean_wav, lens, reduction="batch"
                    )
                elif self.hparams.target_metric == "pesq":
                    self.pesq_metric.append(
                        batch.id, predict=predict_wav, target=clean_wav, lengths=lens
                    )

            # Write wavs to file, for evaluation
            elif self.hparams.mode == 'test':
                self.stoi_metric.append(
                        batch.id, predict_wav, clean_wav, lens, reduction="batch"
                    )
                self.pesq_metric.append(
                        batch.id, predict=predict_wav, target=clean_wav, lengths=lens
                    )
                lens = lens * clean_wav.shape[1]
                for name, pred_wav, length in zip(batch.id, predict_wav, lens):
                    name += ".wav"
                    enhance_path = os.path.join(self.hparams.enhanced_folder, name)
                    torchaudio.save(
                        enhance_path,
                        torch.unsqueeze(pred_wav[: int(length)].cpu(), 0),
                        16000,
                    )

        # we do not use mse_cost to update model
        return adv_cost

    def compute_ids(self, batch_id, optim_name):
        """Returns the list of ids, edited via optimizer name."""
        if optim_name == "D_enh":
            return [f"{uid}@{self.epoch}" for uid in batch_id]
        return batch_id

    def save_noisy_scores(self, batch_id, scores):
        for i, score in zip(batch_id, scores):
            self.noisy_scores[i] = score

    def score(self, batch_id, deg_wav, ref_wav, lens):
        """Returns actual metric score, either pesq or stoi

        Arguments
        ---------
        batch_id : list of str
            A list of the utterance ids for the batch
        deg_wav : torch.Tensor
            The degraded waveform to score
        ref_wav : torch.Tensor
            The reference waveform to use for scoring
        length : torch.Tensor
            The relative lengths of the utterances
        """
        new_ids = [
            i
            for i, d in enumerate(batch_id)
            if d not in self.historical_set and d not in self.noisy_scores
        ]

        if len(new_ids) == 0:
            pass
        elif self.hparams.target_metric == "pesq":
            self.target_metric.append(
                ids=[batch_id[i] for i in new_ids],
                predict=deg_wav[new_ids].detach(),
                target=ref_wav[new_ids].detach(),
                lengths=lens[new_ids],
            )
            score = torch.tensor(
                [[s] for s in self.target_metric.scores], device=self.device,
            )
        elif self.hparams.target_metric == "stoi":
            self.target_metric.append(
                [batch_id[i] for i in new_ids],
                deg_wav[new_ids],
                ref_wav[new_ids],
                lens[new_ids],
                reduction="batch",
            )
            score = torch.tensor(
                [[-s] for s in self.target_metric.scores], device=self.device,
            )
        else:
            raise ValueError("Expected 'pesq' or 'stoi' for target_metric")

        # Clear metric scores to prepare for next batch
        self.target_metric.clear()

        # Combine old scores and new
        final_score = []
        for i, d in enumerate(batch_id):
            if d in self.historical_set:
                final_score.append([self.historical_set[d]["score"]])
            elif d in self.noisy_scores:
                final_score.append([self.noisy_scores[d]])
            else:
                final_score.append([score[new_ids.index(i)]])

        return torch.tensor(final_score, device=self.device)

    def est_score(self, deg_spec, ref_spec):
        """Returns score as estimated by discriminator

        Arguments
        ---------
        deg_spec : torch.Tensor
            The spectral features of the degraded utterance
        ref_spec : torch.Tensor
            The spectral features of the reference utterance
        """
        combined_spec = torch.cat(
            [deg_spec.unsqueeze(1), ref_spec.unsqueeze(1)], 1
        )
        return self.modules.discriminator(combined_spec)

    def write_wavs(self, clean_id, batch_id, wavs, scores, lens):
        """Write wavs to files, for historical discriminator training

        Arguments
        ---------
        batch_id : list of str
            A list of the utterance ids for the batch
        wavs : torch.Tensor
            The wavs to write to files
        score : torch.Tensor
            The actual scores for the corresponding utterances
        lens : torch.Tensor
            The relative lengths of each utterance
        """
        lens = lens * wavs.shape[1]
        record = {}
        for i, (cleanid, name, pred_wav, length) in enumerate(
            zip(clean_id, batch_id, wavs, lens)
        ):
            path = os.path.join(self.hparams.MetricGAN_folder, name + ".wav")
            data = torch.unsqueeze(pred_wav[: int(length)].cpu(), 0)
            torchaudio.save(path, data, self.hparams.Sample_rate)

            # Make record of path and score for historical training
            score = float(scores[i][0])
            clean_path = cleanid.split('-', 1)
            clean_path = os.path.join(
                self.hparams.train_clean_folder, clean_path[0], clean_path[1] + ".pkl"
            )
            record[name] = {
                "enh_wav": path,
                "score": score,
                "clean_wav": clean_path,
            }

        # Update records for historical training
        self.historical_set.update(record)

    def fit_batch(self, batch):
        "Compute gradients and update either D or G based on sub-stage."
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss_tracker = 0
        if self.sub_stage == SubStage.CURRENT:
            for mode in ["clean", "enh", "noisy"]:
                loss = self.compute_objectives(
                    predictions, batch, sb.Stage.TRAIN, f"D_{mode}"
                )
                self.d_optimizer.zero_grad()
                loss.backward()
                if self.check_gradients(loss):
                    self.d_optimizer.step()
                loss_tracker += loss.detach() / 3
        elif self.sub_stage == SubStage.HISTORICAL:
            loss = self.compute_objectives(
                predictions, batch, sb.Stage.TRAIN, "D_enh"
            )
            self.d_optimizer.zero_grad()
            loss.backward()
            if self.check_gradients(loss):
                self.d_optimizer.step()
            loss_tracker += loss.detach()
        elif self.sub_stage == SubStage.GENERATOR:
            loss = self.compute_objectives(
                predictions, batch, sb.Stage.TRAIN, "generator"
            )
            self.g_optimizer.zero_grad()
            loss.backward()
            if self.check_gradients(loss):
                self.g_optimizer.step()
            loss_tracker += loss.detach()
        return loss_tracker

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch

        This method calls ``fit()`` again to train the discriminator
        before proceeding with generator training.
        """

        self.mse_metric = MetricStats(metric=self.hparams.compute_cost)
        self.metrics = {"G": [], "D": []}

        if stage == sb.Stage.TRAIN:
            if self.hparams.target_metric == "pesq":
                self.target_metric = MetricStats(metric=pesq_eval, n_jobs=40)
            elif self.hparams.target_metric == "stoi":
                self.target_metric = MetricStats(metric=stoi_loss)
            else:
                raise NotImplementedError(
                    "Right now we only support 'pesq' and 'stoi'"
                )

            # Train discriminator before we start generator training
            if self.sub_stage == SubStage.GENERATOR:
                self.epoch = epoch
                self.train_discriminator()
                self.sub_stage = SubStage.GENERATOR
                print("Generator training by current data...")

        if stage != sb.Stage.TRAIN:
            self.pesq_metric = MetricStats(metric=pesq_eval, n_jobs=30)
            self.stoi_metric = MetricStats(metric=stoi_loss)

    def train_discriminator(self):
        """A total of 3 data passes to update discriminator."""
        # First, iterate train subset w/ updates for clean, enh, noisy
        print("Discriminator training by current data...")
        self.sub_stage = SubStage.CURRENT
        self.fit(
            range(1),
            self.train_set,
            train_loader_kwargs=self.hparams.dataloader_options,
        )

        # Next, iterate historical subset w/ updates for enh
        if self.historical_set:
            print("Discriminator training by historical data...")
            self.sub_stage = SubStage.HISTORICAL
            self.fit(
                range(1),
                self.historical_set,
                train_loader_kwargs=self.hparams.dataloader_options,
            )

        # Finally, iterate train set again. Should iterate same
        # samples as before, due to ReproducibleRandomSampler
        print("Discriminator training by current data again...")
        self.sub_stage = SubStage.CURRENT
        self.fit(
            range(1),
            self.train_set,
            train_loader_kwargs=self.hparams.dataloader_options,
        )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        "Called at the end of each stage to summarize progress"
        # epoch is awared in each stage
        def ckpt_predicate(ckpt):
            return ckpt.meta['epoch'] == epoch
        
        def ckpt_predicate_lessthan(ckpt):
            return ckpt.meta['epoch'] < epoch

        if self.sub_stage != SubStage.GENERATOR:
            return
        
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            g_loss = torch.tensor(self.metrics["G"])  # batch_size
            d_loss = torch.tensor(self.metrics["D"])  # batch_size
            print("Avg G loss: %.3f" % torch.mean(g_loss))
            print("Avg D loss: %.3f" % torch.mean(d_loss))
            print("MSE distance: %.3f" % self.mse_metric.summarize("average"))
            # save the checkpoint every 10 epochs
            # use default timestamp or use epoch numbers as max key
            # run the test as 
            stats = {
                "epoch": epoch,
                "MSE distance": stage_loss,
                # "target_metric": self.target_metric.summarize("average")
            }
            self.checkpointer.save_checkpoint(meta=stats)

        elif self.hparams.target_metric == "pesq":
            stats = {
                "epoch": epoch,
                "MSE distance": stage_loss,
                "pesq": 5 * self.pesq_metric.summarize("average") - 0.5,
            }
        elif self.hparams.target_metric == "stoi":
            stats = {
                "epoch": epoch,
                "MSE distance": stage_loss,
                "stoi": -self.stoi_metric.summarize("average"),
            }

        if stage == sb.Stage.VALID:
            if self.hparams.use_tensorboard:
                valid_stats = {
                    "mse": stage_loss,
                    "pesq": 5 * self.pesq_metric.summarize("average") - 0.5,
                    "stoi": -self.stoi_metric.summarize("average"),
                }
                self.hparams.tensorboard_train_logger.log_stats(valid_stats)
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stats, max_keys=[self.hparams.target_metric]
            )

        if stage == sb.Stage.TEST:
            if self.hparams.mode == 'val':
                ckpts = self.checkpointer.find_checkpoints(ckpt_predicate=ckpt_predicate)
                assert len(ckpts) == 1
                # delete old ckpt from train
                self.checkpointer.delete_checkpoints(num_to_keep=0, ckpt_predicate=ckpt_predicate)

                if self.hparams.use_tensorboard:
                    valid_stats = stats
                    # will two tensorboard corrupt?
                    self.hparams.tensorboard_train_logger.log_stats(valid_stats)
                self.hparams.train_logger.log_stats(
                    {"Epoch": epoch},
                    valid_stats=stats,
                )
                # TODO:save current checkpointer again
                self.checkpointer.save_and_keep_only(
                    meta=stats, max_keys=[self.hparams.target_metric], ckpt_predicate=ckpt_predicate_lessthan
                    )
                # self.checkpointer.save_checkpoint(meta=stats)

            else:
                print("Epoch loaded", self.hparams.epoch_counter.current)
                print("stoi", -self.stoi_metric.summarize("average"))
                print("pesq", 5 * self.pesq_metric.summarize("average") - 0.5)
                test_stats = {
                    "mse": stage_loss,
                    "pesq": 5 * self.pesq_metric.summarize("average") - 0.5,
                    "stoi": -self.stoi_metric.summarize("average"),
                }
                # self.hparams.train_logger.log_stats(
                #     {"Epoch loaded": self.hparams.epoch_counter.current},
                #     test_stats=test_stats,
                # )

    def make_dataloader(
        self, dataset, stage, ckpt_prefix="dataloader-", **loader_kwargs
    ):
        "Override dataloader to insert custom sampler/dataset"
        if stage == sb.Stage.TRAIN:

            # Create a new dataset each time, this set grows
            if self.sub_stage == SubStage.HISTORICAL:
                dataset = sb.dataio.dataset.DynamicItemDataset(
                    data=dataset,
                    dynamic_items=[enh_pipeline],
                    output_keys=["id", "enh_sig", "clean_sig", "score"],
                )
                samples = round(len(dataset) * self.hparams.history_portion)
            else:
                samples = self.hparams.number_of_samples

            # This sampler should give the same samples for D and G
            epoch = self.hparams.epoch_counter.current

            # Equal weights for all samples, we use "Weighted" so we can do
            # both "replacement=False" and a set number of samples, reproducibly
            weights = torch.ones(len(dataset))
            sampler = ReproducibleWeightedRandomSampler(
                weights, epoch=epoch, replacement=False, num_samples=samples
            )
            loader_kwargs["sampler"] = sampler

            if self.sub_stage == SubStage.GENERATOR:
                self.train_sampler = sampler

        # Make the dataloader as normal
        return super().make_dataloader(
            dataset, stage, ckpt_prefix, **loader_kwargs
        )

    def on_fit_start(self):
        "Override to prevent this from running for D training"
        if self.sub_stage == SubStage.GENERATOR:
            super().on_fit_start()

    def init_optimizers(self):
        "Initializes the generator and discriminator optimizers"
        self.g_optimizer = self.hparams.g_opt_class(
            self.modules.generator.parameters()
        )
        self.d_optimizer = self.hparams.d_opt_class(
            self.modules.discriminator.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("g_opt", self.g_optimizer)
            self.checkpointer.add_recoverable("d_opt", self.d_optimizer)


# Define audio piplines
@sb.utils.data_pipeline.takes("pkl")
@sb.utils.data_pipeline.provides("noisy_sig", "clean_sig")
def audio_pipeline(pkl_path):
    with open(pkl_path, 'rb') as fo:
        data = pickle.load(fo)
        yield data[0][:len(data[1])]/torch.max(torch.abs(data[0][:len(data[1])]))
        yield data[1]/torch.max(torch.abs(data[1]))

# For historical data
@sb.utils.data_pipeline.takes("enh_wav", "clean_wav")
@sb.utils.data_pipeline.provides("enh_sig", "clean_sig")
def enh_pipeline(enh_wav, clean_wav):
    yield sb.dataio.dataio.read_audio(enh_wav)
    with open(clean_wav, 'rb') as fo:
        data = pickle.load(fo)
        # yield data[1]
        yield data[1]/torch.max(torch.abs(data[1]))

# @sb.utils.data_pipeline.takes("pkl")
# @sb.utils.data_pipeline.provides("noisy_sig", "clean_sig", "loc", "images", "ra")
# def multimodal_pipeline(pkl_path):
#     with open(pkl_path, 'rb') as fo:
#         data = pickle.load(fo)
#         yield data[0][:len(data[1])]/torch.max(torch.abs(data[0][:len(data[1])]))
#         yield data[1]/torch.max(torch.abs(data[1]))


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class."""

    # Define datasets
    datasets = {}
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["id", "noisy_sig", "clean_sig"],
        )

    return datasets


def create_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


# Recipe begins!
if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Data preparation
    # from voicebank_prepare import prepare_voicebank  # noqa

    # run_on_main(
    #     prepare_voicebank,
    #     kwargs={
    #         "data_folder": hparams["data_folder"],
    #         "save_folder": hparams["save_folder"],
    #         "skip_prep": hparams["skip_prep"],
    #     },
    # )

    # Create dataset objects
    datasets = dataio_prep(hparams)
    print(datasets['test'])
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger

        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs"]
        )

    # Create the folder to save enhanced files (+ support for DDP)
    run_on_main(create_folder, kwargs={"folder": hparams["enhanced_folder"]})

    se_brain = MetricGanBrain(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    se_brain.train_set = datasets["train"]
    se_brain.historical_set = {}
    se_brain.noisy_scores = {}
    se_brain.batch_size = hparams["dataloader_options"]["batch_size"]
    se_brain.valid_batch_size = hparams["valid_dataloader_options"]["batch_size"]
    se_brain.sub_stage = SubStage.GENERATOR

    # shutil.rmtree(hparams["MetricGAN_folder"])
    run_on_main(create_folder, kwargs={"folder": hparams["MetricGAN_folder"]})

    # Load latest checkpoint to resume training

    # do not valid in train process

    if hparams["mode"] == "train":
        se_brain.fit(
            epoch_counter=se_brain.hparams.epoch_counter,
            train_set=datasets["train"],
            # valid_set=datasets["valid"],
            train_loader_kwargs=hparams["dataloader_options"],
            # valid_loader_kwargs=hparams["valid_dataloader_options"],
        )
    elif hparams["mode"] == "val":
        # evaluate is totally same as valid except requiring loading the ckpt by max keys
        # use ckpt_predicate to filter epoch == specific one
        # overwrite current checkpoint with the new max_key of pesq/stoi
        epoch_init = 362
        def ckpt_predicate(ckpt):
            return ckpt.meta['epoch'] == epoch_init
        while epoch_init <= hparams['number_of_epochs']:
            ckpt = se_brain.checkpointer.find_checkpoint(ckpt_predicate=ckpt_predicate)
            if ckpt is not None:
                print(f"validing epoch {epoch_init}")
                test_stats = se_brain.evaluate(
                    test_set=datasets["valid"],
                    ckpt_predicate=ckpt_predicate,
                    test_loader_kwargs=hparams["valid_dataloader_options"],
                    epoch=epoch_init,
                    )
                epoch_init += hparams['valid_frequency']
            else:
                print(f"waiting for new checpoints of epoch {epoch_init}")
                time.sleep(30)

    elif hparams["mode"] == "test":
        # Load best checkpoint for evaluation
        
        test_stats = se_brain.evaluate(
            test_set=datasets["test"],
            max_key=hparams["target_metric"],
            test_loader_kwargs=hparams["valid_dataloader_options"],
        )

    # run only the test, comment the others
    # generated audio, use the generated audio to compute ASR and compute SISNR
    # use the same wave as previous ASR-setup, each test-clean will be transcribe once.
    # 