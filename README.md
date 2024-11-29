This repo contains training files and commands

The models I want to train:

<ol>
<li> VITS 
    <ul>
    <li> With Phoneme </li>
    <li> <strike> Without Phoneme </strike> ✅ </li>
    </ul>
</li>
<li> GlowTTS + HiFiGAN++ 
    <ul>
    <li> With Phoneme </li>
    <li> <strike> Without Phoneme </strike> ✅ </li>
    </ul>
</li>
<li> Tacotron 2 
    <ul>
    <li> With Phoneme </li>
    <li> Without Phoneme </li>
    </ul>
</li>
<li> FastSpeech
    <ul>
    <li> With Phoneme </li>
    <li> Without Phoneme </li>
    </ul>
</li>
</ol>

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


<h2> <i>Update</i> </h2>

**EDA** -- [Google Sheet](https://docs.google.com/spreadsheets/d/1dEBrH9h0dPXl7ePmS5y18KAhokVU37q9J8Y9ZT6g15w/edit#gid=1353322539)

<h2>To Do</h2>
<ul>
<li>Audio Duration ✅</li>
<li>Character list ✅</li>
<li>Phoneme list ✅</li>
<li>Character Frequency ✅</li>
<li>Word Frequency ✅</li>
<li>Word to Phoneme ✅</li>
<li>Character to Phoneme</li>
<li>Sentence to Phoneme ✅</li>
</ol>