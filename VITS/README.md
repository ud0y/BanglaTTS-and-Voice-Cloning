To train VITS, we can do it with `train.py` script and with `train.ipynb` file.

`train.py` argument list:

`--meta_file` Set directory to load meta data, e.g., "mono/metadata_male.txt" <br>
default="content/iitm_bangla_tts/comprehensive_bangla_tts/male/mono/metadata_male.txt" <br>

`--root_path` Set directory to load meta data, e.g., "mono" <br>
default="content/iitm_bangla_tts/comprehensive_bangla_tts/male/mono" <br>

`--output_path` Set directory to save others, e.g., "cache" <br>
default="" <br>

`--sample_rate` Set sample rate, e.g., 16000 <br>
default=22050 <br>

`--win_length` Set window length, e.g., 1000 <br>
default=1024 <br>

`--hop_length` Set hop length, e.g., 128 <br>
default=256 <br>

`--num_mels` Set number of melspectrogram, e.g., 100 <br>
default=80 <br>

`--mel_fmin` Set fmin for melspectrogram, e.g., 1 <br>
default=0 <br>

`--mel_fmax` Set fmax for melspectrogram, e.g., None <br>
default=None <br>

`--run_name` Set directory to store checkpoint, e.g., "VITS" <br>
default="Vits" <br>

`--batch_size` Set batch size, e.g., 8 <br>
default=16 <br>

`--eval_batch_size` Set evaluation batch size, e.g., 8 <br>
default=8 <br>

`--epochs` Set number of epochs, e.g., 10000 <br>
default=1500 <br>

`--pretrained_path` Set directory to load pretrained model, e.g., "VITS" <br>
default="" <br>

`--num_loader_workers` Set number of CPUs you have, e.g., 16 <br>
default=8 <br>

`--num_eval_loader_workers` Set number of CPUs you have, e.g., 16 <br>
default=8 <br>

`--save_step` Set number of steps after that model checkpoints will be saved, e.g., 1000 <br>
default=10000<br>

`--print_step` Set number of steps after that model status will be printed, e.g., 1000 <br>
default=100 <br>

To train with `train.py` script type following command in conda environment:
```
python train.py --epochs 10000 --batch_size 48 --eval_batch_size 32 --output_path vits_male
```
