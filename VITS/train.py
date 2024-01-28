import argparse
import os
import gdown
import pandas as pd
import soundfile as sf
import shutil
import bangla
import torch
import pysbd
import numpy as np

try:
    from TTS.utils.synthesizer import Synthesizer
except:
    print("coundn't import TTS synthesizer,trying again!")

import TTS
from typing import List

from TTS.config import load_config
from TTS.tts.models import setup_model as setup_tts_model

from TTS.tts.utils.synthesis import synthesis, transfer_voice, trim_silence
from TTS.utils.audio import AudioProcessor
from TTS.vocoder.models import setup_model as setup_vocoder_model
from TTS.vocoder.utils.generic_utils import interpolate_vocoder_input
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import (
    BaseDatasetConfig,
    BaseAudioConfig,
    CharactersConfig,
)
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.speakers import SpeakerManager
import time

# Get the command line input into the scripts
parser = argparse.ArgumentParser()

parser.add_argument(
    "--meta_file",
    action="store",
    default="content/iitm_bangla_tts/comprehensive_bangla_tts/male/mono/metadata_male.txt",
    dest="meta_file",
    help='Set directory to load meta data, e.g., "mono/metadata_male.txt"',
)

parser.add_argument(
    "--root_path",
    action="store",
    default="content/iitm_bangla_tts/comprehensive_bangla_tts/male/mono",
    dest="root_path",
    help='Set directory to load meta data, e.g., "mono"',
)

parser.add_argument(
    "--output_path",
    action="store",
    default="",
    dest="output_path",
    help='Set directory to load meta data, e.g., "saved_model"',
)

# configuration
parser.add_argument(
    "--sample_rate",
    action="store",
    default=22050,
    dest="sample_rate",
    help="Set directory to load meta data, e.g., 16000",
)

parser.add_argument(
    "--win_length",
    action="store",
    default=1024,
    dest="win_length",
    help="Set directory to load meta data, e.g., 1000",
)

parser.add_argument(
    "--hop_length",
    action="store",
    default=256,
    dest="hop_length",
    help="Set directory to load meta data, e.g., 128",
)

parser.add_argument(
    "--num_mels",
    action="store",
    default=80,
    dest="num_mels",
    help="Set directory to load meta data, e.g., 100",
)

parser.add_argument(
    "--mel_fmin",
    action="store",
    default=0,
    dest="mel_fmin",
    help="Set directory to load meta data, e.g., 1",
)

parser.add_argument(
    "--mel_fmax",
    action="store",
    default=None,
    dest="mel_fmax",
    help="Set directory to load meta data, e.g., None",
)

parser.add_argument(
    "--run_name",
    action="store",
    default="Vits",
    dest="run_name",
    help="Set directory to load meta data, e.g., GlowTTS",
)

parser.add_argument(
    "--batch_size",
    action="store",
    default=16,
    dest="batch_size",
    help="Set directory to load meta data, e.g., 8",
)

parser.add_argument(
    "--eval_batch_size",
    action="store",
    default=8,
    dest="eval_batch_size",
    help="Set directory to load meta data, e.g., 8",
)

parser.add_argument(
    "--epochs",
    action="store",
    default=1500,
    dest="epochs",
    help="Set directory to load meta data, e.g., 100",
)

parser.add_argument(
    "--pretrained_path",
    action="store",
    default="",
    dest="pretrained_path",
    help='Set directory to load meta data, e.g., "Vits_path"',
)

parser.add_argument(
    "--num_loader_workers",
    action="store",
    default=8,
    dest="num_loader_workers",
    help="Set number of CPUs you have, e.g., 16",
)

parser.add_argument(
    "--num_eval_loader_workers",
    action="store",
    default=16,
    dest="num_eval_loader_workers",
    help="Set number of CPUs you have, e.g., 16",
)

parser.add_argument(
    "--save_step",
    action="store",
    default=10000,
    dest="save_step",
    help="Set number of steps after that model checkpoints will be saved, e.g., 1000",
)

parser.add_argument(
    "--print_step",
    action="store",
    default=100,
    dest="print_step",
    help="Set number of steps after that model status will be printed, e.g., 1000",
)


def formatter(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalizes the LJSpeech meta data file to TTS format
    https://keithito.com/LJ-Speech-Dataset/"""
    txt_file = meta_file
    items = []
    speaker_name = "ljspeech"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wav", cols[0] + ".wav")
            try:
                text = cols[1]
            except:
                print("not found")

            items.append(
                {
                    "text": text,
                    "audio_file": wav_file,
                    "speaker_name": speaker_name,
                    "root_path": root_path,
                }
            )
    return items


## parse the results
parse_results = parser.parse_args()

meta_file = parse_results.meta_file
root_path = parse_results.root_path
output_path = parse_results.output_path

# model parameters
sample_rate = int(parse_results.sample_rate)
win_length = int(parse_results.win_length)
hop_length = int(parse_results.hop_length)
num_mels = int(parse_results.num_mels)
mel_fmin = int(parse_results.mel_fmin)
mel_fmax = parse_results.mel_fmax
# training parameters
run_name = parse_results.run_name
batch_size = int(parse_results.batch_size)
eval_batch_size = int(parse_results.eval_batch_size)
epochs = int(parse_results.epochs)
pretrained_path = parse_results.pretrained_path
num_loader_workers = int(parse_results.num_loader_workers)
num_eval_loader_workers = int(parse_results.num_eval_loader_workers)
save_step = int(parse_results.save_step)
print_step = int(parse_results.print_step)

# define the dataset
dataset_config = BaseDatasetConfig(
    meta_file_train=meta_file, path=os.path.join(root_path, "")
)
train_samples, eval_samples = load_tts_samples(
    dataset_config, formatter=formatter, eval_split=True
)

audio_config = VitsAudioConfig(
    sample_rate=sample_rate,
    win_length=win_length,
    hop_length=hop_length,
    num_mels=num_mels,
    mel_fmin=mel_fmin,
    mel_fmax=mel_fmax,
)

if "male" in meta_file:
    characters_config = CharactersConfig(
        pad="<PAD>",
        eos="।",  #'<EOS>', #'।',
        bos="<BOS>",  # None,
        blank="<BLNK>",
        phonemes=None,
        characters="তট৫ভিঐঋখঊড়ইজমএেঘঙসীঢ়হঞ‘ঈকণ৬ঁৗশঢঠ\u200c১্২৮দৃঔগও—ছউংবৈঝাযফ\u200dচরষঅৌৎথড়৪ধ০ুূ৩আঃপয়’নলো",
        punctuations="-!,|.? ",
    )
else:
    characters_config = CharactersConfig(
        pad="<PAD>",
        eos="।",  #'<EOS>', #'।',
        bos="<BOS>",  # None,
        blank="<BLNK>",
        phonemes=None,
        characters="ইগং়’ুঃন১ঝূও‘ঊোছপফৈ৮ষযৎঢঈকঠিজ০৬ীটডএঅঋধচে২৩ণউয়ঢ়খলভৗসহ্ড়দথবঔাঞশরৌম—ঐআৃঘঙ\u200cঁ৪৫ত",
        punctuations=".?-!|, ",
    )

# VitsConfig: all model related values for training, validating and testing.
config = VitsConfig(
    audio=audio_config,
    run_name=run_name,
    batch_size=batch_size,
    eval_batch_size=eval_batch_size,
    batch_group_size=0,
    num_loader_workers=num_loader_workers,
    num_eval_loader_workers=num_eval_loader_workers,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=epochs,
    text_cleaner=None,
    use_phonemes=True,
    phoneme_language="bn",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=print_step,
    print_eval=False,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    characters=characters_config,
    save_step=save_step,
    cudnn_benchmark=True,
    test_sentences=[
        "হয়,হয়ে,ওয়া,হয়েছ,হয়েছে,দিয়ে,যায়,দায়,নিশ্চয়,আয়,ভয়,নয়,আয়াত,নিয়ে,হয়েছে,দিয়েছ,রয়ে,রয়েছ,রয়েছে।",
        "দেয়,দেওয়া,বিষয়,হয়,হওয়া,সম্প্রদায়,সময়,হয়েছি,দিয়েছি,হয়,হয়েছিল,বিষয়ে,নয়,কিয়াম,ইয়া,দেয়া,দিয়েছে,আয়াতে,দয়া।",
        "ইয়াহুদ,নয়,ব্যয়,ইয়াহুদী,নেওয়া,উভয়ে,যায়,হয়েছিল,প্রয়োজন।",
    ],
)

ap = AudioProcessor.init_from_config(config)

tokenizer, config = TTSTokenizer.init_from_config(config)

model = Vits(config, ap, tokenizer, speaker_manager=None)

trainer = Trainer(
    TrainerArgs(continue_path=pretrained_path),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

# let's 🚀
trainer.fit()
