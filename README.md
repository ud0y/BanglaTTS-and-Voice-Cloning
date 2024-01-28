This repo contains training files and commands

The models I want to train:

1. VITS (with and without phoneme)
2. GlowTTS + HiFiGAN++ (with and without phoneme)
3. Tacotron 2 (with and without phoneme)
4. FastSpeech (with and without phoneme)

To get started follow below instructions,

1. Create a new conda environment, <br>
   ```
    conda create -n tts python=3.10
   ``` 
    Now activate the environment, <br>
    ```
   conda activate tts
    ```

3. Install epseask(-ng): This will be used for phoneme <br>
   ```
   sudo apt-get install espeak-ng -y
   ```
    If this does not work, try <br>
   ```
   sudo apt-get install espeak -y
   ```

5. Now clone the Coqui/TTS directory, <br>
    ```
   git clone https://github.com/coqui-ai/TTS.git
    ```

7. Install the dependencies, <br>
    ```
   pip install -e TTS[all,dev,notebooks]
    ```

9. Now download the dataset from google drive <br>
    ```
    pip install gdown
    gdown 1QLCfoO_2In5AIJtMzvVEmB-_TnUwtSUb
    unzip dataset.zip
    ```
