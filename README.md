<style>
  /* Define CSS for list items */
  ul {
    list-style-type: none;
    padding: 0;
  }
  li::before {
    content: attr(data-emoji); /* Use content attribute to display emoji */
    margin-right: 5px; /* Add some spacing between emoji and text */
  }
</style>

This repo contains training files and commands

The models I want to train:

<ol>
<li> VITS 
    <ul>
    <li data-emoji="⭕"> With Phoneme </li>
    <li data-emoji="✅"> Without Phoneme</li>
    </ul>
</li>
<li> GlowTTS + HiFiGAN++ 
    <ul>
    <li data-emoji="⭕"> With Phoneme </li>
    <li data-emoji="⭕"> Without Phoneme </li>
    </ul>
</li>
<li> Tacotron 2 
    <ul>
    <li data-emoji="⭕"> With Phoneme </li>
    <li data-emoji="⭕"> Without Phoneme </li>
    </ul>
</li>
<li> FastSpeech
    <ul>
    <li data-emoji="⭕"> With Phoneme </li>
    <li data-emoji="⭕"> Without Phoneme </li>
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
<li data-emoji="✅">Audio Duration</li>
<li data-emoji="✅">Character list</li>
<li data-emoji="✅">Phoneme list</li>
<li data-emoji="✅">Character Frequency</li>
<li data-emoji="✅">Word Frequency</li>
<li data-emoji="✅">Word to Phoneme</li>
<li data-emoji="⛔">Character to Phoneme</li>
<li data-emoji="✅">Sentence to Phoneme</li>
</ol>