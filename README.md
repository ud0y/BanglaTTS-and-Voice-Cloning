This repo contains training files and commands

The models I want to train:

    1. VITS (with and without phoneme)
    2. GlowTTS + HiFiGAN++ (with and without phoneme)
    3. Tacotron 2 (with and without phoneme)
    4. FastSpeech (with and without phoneme)

To get started follow below instructions,

1. Create a new conda environment,
    `conda create -n tts python=3.10`
    Now activate the environment,
    `conda activate tts`

2. Install epseask(-ng): This will be used for phoneme
    `sudo apt-get install espeak-ng -y`
    If this does not work, try
    `sudo apt-get install espeak -y`

3. Now clone the Coqui/TTS directory,
    `git clone https://github.com/coqui-ai/TTS.git`

4. Install the dependencies,
    `pip install -e TTS[all,dev,notebooks]`

5. Now download the dataset from google drive
    `pip install gdown`
    `gdown 1QLCfoO_2In5AIJtMzvVEmB-_TnUwtSUb`
    `unzip dataset.zip`