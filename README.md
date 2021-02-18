# Speaker Verification with [NeMo](https://github.com/NVIDIA/NeMo)
*This is merely an experiment, the dataset `an4` used for the experiment is not suitable for training a speaker verification model, it was only usde for fast training.*
Nemo's examples and tutorials do not provide explicit illustrations to speaker verification. This will be an experiment of Speaker verification with NeMo.

## Data Preparation
*This step has been done in a previous experiment, manifest files were directly moved to the data folder of this repo. The following describes how manifests were obtained.*<br>
Run `download_and_convert_an4.py` to download dataset `an4`, and convert `.sph` to `.wav`.<br>
Generate manifest files:<br>
`find {data_dir}/an4/wav/an4_clstk  -iname "*.wav" > data/an4/wav/an4_clstk/train_all.scp`<br>
preview<br>
`head -n 3 {data_dir}/an4/wav/an4_clstk/train_all.scp`<br>

Convert `.scp` to `manifest`, set the --split flag for splitting training and development set:<br>
`python {path-to/scp_to_manifest.py} --scp {paths-to/train_all.scp} --id -2 --out {path-to-opt/all_manifest.json} --split`<br>

`scp_to_manifest.py` src: [https://github.com/NVIDIA/NeMo/blob/main/scripts/scp_to_manifest.py](https://github.com/NVIDIA/NeMo/blob/main/scripts/scp_to_manifest.py)


## Speaker Verification Model Training
configuration file src: [https://github.com/NVIDIA/NeMo/blob/main/examples/speaker_recognition/conf/SpeakerNet_verification_3x2x512.yaml](https://github.com/NVIDIA/NeMo/blob/main/examples/speaker_recognition/conf/SpeakerNet_verification_3x2x512.yaml)<br>
Run `scripts/train_spk_ver_model.py` to train a speaker verification model.

## Speaker Embeddings Extraction
First generate `embeddings_manifest.json` for test. For the purpose of this experiment I created this manifest manually.<br>
Run `get_embs.py`.<br>
As the purpose of this experiment is to verify my voice, the recordings were of me, the recordings are not uploaded in this repo. (Audios for test were recorded on `Praat`.)

## Speaker Verification: cosine-similarity of embeddings
Calculate cosine-similarity of the two speaker embeddings to see the certainty of this model of two audios being from the same speaker.<br>

*A first experiment score: 0.9686643297704525*

