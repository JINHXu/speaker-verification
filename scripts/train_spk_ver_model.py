# train a speaker verification model with NeMo
# dataset: an4, for fast training

# Jinghua Xu


import os
import torch

import pytorch_lightning as pl
from omegaconf.listconfig import ListConfig
from pytorch_lightning import seed_everything

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

seed_everything(42)


@hydra_runner(config_path="/Users/xujinghua/NeMo/examples/speaker_recognition/conf", config_name="SpeakerNet_verification_3x2x512.yaml")
def main(cfg):

    # add paths to manifests to config
    cfg.model.train_ds.manifest_filepath = '/Users/xujinghua/speaker-verification-with-NeMo/data/train.json'
    cfg.model.validation_ds.manifest_filepath = '/Users/xujinghua/speaker-verification-with-NeMo/data/train.json'

    # an4 test files have a different set of speakers
    # cfg.model.test_ds.manifest_filepath = '/Users/xujinghua/NeMo/data/an4/wav/an4_clstk/dev.json'

    cfg.model.decoder.num_classes = 74

    os.environ["OMP_NUM_THREADS"] = '1'

    # tutorial default setting: flags
    # modify some trainer configs for this demo
    # Checks if we have GPU available and uses it
    cuda = 1 if torch.cuda.is_available() else 0
    cfg.trainer.gpus = cuda

    # Reduces maximum number of epochs to 5 for quick demonstration
    cfg.trainer.max_epochs = 5

    # Remove distributed training flags
    cfg.trainer.accelerator = None

    logging.info(f'Hydra config: {cfg.pretty()}')
    trainer = pl.Trainer(**cfg.trainer)
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    speaker_model = EncDecSpeakerLabelModel(cfg=cfg.model, trainer=trainer)
    trainer.fit(speaker_model)

    if not trainer.fast_dev_run:
        model_path = os.path.join(log_dir, '..', 'spkr.nemo')
        speaker_model.save_to(model_path)

    # no need for testing

    '''
    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        gpu = 1 if cfg.trainer.gpus != 0 else 0
        trainer = pl.Trainer(gpus=gpu)
        if speaker_model.prepare_test(trainer):
            result = trainer.test(speaker_model)
            # , ckpt_path='/Users/xujinghua/NeMo/data/an4/wav/an4_clstk/dev.json')
            
            print(result)
    '''


if __name__ == '__main__':
    main()
