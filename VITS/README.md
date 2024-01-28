To train VITS, we can do it with `train.py` script and with `train.ipynb` file.

`train.py` argument list:

    `--meta_file`
    default="content/iitm_bangla_tts/comprehensive_bangla_tts/male/mono/metadata_male.txt"
    help='Set directory to load meta data, e.g., "mono/metadata_male.txt"'

    `--root_path`
    default="content/iitm_bangla_tts/comprehensive_bangla_tts/male/mono"
    help='Set directory to load meta data, e.g., "mono"'

    `--output_path`
    default=""
    help='Set directory to load meta data, e.g., "saved_model"'

    `--sample_rate`
    default=22050,
    help="Set directory to load meta data, e.g., 16000"

    `--win_length`
    default=1024
    help="Set directory to load meta data, e.g., 1000"

    `--hop_length`
    default=256
    help="Set directory to load meta data, e.g., 128"

    `--num_mels`
    default=80
    help="Set directory to load meta data, e.g., 100"

    `--mel_fmin`
    default=0
    help="Set directory to load meta data, e.g., 1"

    `--mel_fmax`
    default=None
    help="Set directory to load meta data, e.g., None"

    `--run_name`
    default="Vits"
    help="Set directory to load meta data, e.g., GlowTTS"

    `--batch_size`
    default=16
    help="Set directory to load meta data, e.g., 8"

    `--eval_batch_size`
    default=8,
    help="Set directory to load meta data, e.g., 8"

    `--epochs`
    default=1500
    help="Set directory to load meta data, e.g., 100"

    `--pretrained_path`
    default=""
    help='Set directory to load meta data, e.g., "Vits_path"'

    `--num_loader_workers`
    default=8 
    help="Set number of CPUs you have, e.g., 16",

    `--num_eval_loader_workers`
    default=16
    help="Set number of CPUs you have, e.g., 16"

    `--save_step`
    default=10000
    help="Set number of steps after that model checkpoints will be saved, e.g., 1000"

    `--print_step`
    default=100
    help="Set number of steps after that model status will be printed, e.g., 1000"

To train with `train.py` script type following command in conda environment:

    `python train.py --epochs 10000 --batch_size 48 --eval_batch_size 32 --output_path vits_male`