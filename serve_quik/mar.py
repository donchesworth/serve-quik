from pathlib import Path
import hugging_quik as hq
from serve_quik import utils
from typing import List, Optional, OrderedDict, KeysView
from argparse import Namespace
import shlex
import shutil
import subprocess
import logging
import sys

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EXTRA_FILES = [
    "config.json",
    "index_to_name.json",
    "setup_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
]
TRANS_FILES = [
    "config.json",
    "setup_config.json",
    "source.spm",
    "special_tokens_map.json",
    "target.spm",
    "tokenizer_config.json",
    "vocab.json",
]


def save_setup_config(
    serve_path: str,
    args: Optional[Namespace] = None,
    labels: Optional[KeysView] = None,
):
    """Create a setup_config json for the huggingface torch serve handler

    Args:
        serve_path (str): The directory to store the sample.
        labels (Optional, KeysView): the dictionary keys of index_labels
        args (Namespace): the project argparse namespace.
    """
    kwargs = getattr(args, "kwargs", {})
    model_type = getattr(args, "model_type", "bert")
    serve_config = hq.model.model_info(model_type, labels, kwargs)
    hq.io.json_write(serve_path, "setup_config.json", serve_config)


def save_index_to_name(serve_path: str, indexed_labels: OrderedDict[int, str]):
    """Create the required index_to_name file for serving

    Args:
        serve_path (str): The directory to store the sample.
        indexed_labels (OrderedDict[str, int]): the target labels with indexes.
    """
    hq.io.json_write(serve_path, "index_to_name.json", indexed_labels)


def save_sample(serve_path):
    """A sample input to test the serving model

    Args:
        serve_path (str): The directory to store the sample.
    """
    sample = ["Great company with fast support"]
    sample = utils.txt_format(sample)
    hq.io.json_write(serve_path, "sample_text.json", sample)


def copy_handler(model_dir: Path, handler_path: Optional[Path] = None):
    """Copy the handler file for serving.

    Args:
        model_dir (Path): The directory with the model files
        handler_path (Path, optional): The location of a custom handler.
        Defaults to None when using text_handler.py
    """
    if handler_path is None:
        handler_path = Path(__file__).parent.joinpath(
            "handler", "text_handler.py"
            )
    shutil.copy(handler_path, model_dir)


def mar_files(mar_path: Path, filelist: List[str]) -> bool:
    """check of all mar files exist before attempting to
    run torch-model-archiver

    Args:
        mar_path (Path): the directory containing all the mar files.
        filelist (List): list of required files for the mar

    Returns:
        bool: Whether all mar files were found.
    """
    missing = [
        file for file in filelist if not mar_path.joinpath(file).is_file()
    ]
    if len(missing) == 0:
        return True
    else:
        logger.info(f"These files are missing: \n{missing}")
        return False


def build_extra_files(
    args: Namespace,
    indexed_labels: OrderedDict[str, int],
    serve_path: Optional[Path] = None,
):
    """build all the boiler plate files to create
    an mar

    Args:
        args (Namespace): the project argparse namespace.
        indexed_labels (OrderedDict[str, int]): the target labels with indexes.
        serve_path (Optional[Path], optional): The diretory to contain
        all serve files. Defaults to None.
    """
    if serve_path is None:
        serve_path = Path.cwd().joinpath("serve")
    serve_path.mkdir(parents=True, exist_ok=True)
    save_setup_config(serve_path, indexed_labels.keys(), args)
    save_index_to_name(serve_path, indexed_labels)
    save_sample(serve_path)
    copy_handler(serve_path)


def create_mar(
    model_type: str,
    model_dir: Path,
    version: Optional[float] = 1.0,
    serialized_file: Optional[str] = None,
    handler: Optional[str] = "text_handler.py",
):
    """build a torch-model-archive file using
    https://github.com/pytorch/serve/tree/master/model-archiver

    Args:
        model_type (str): type of model (e.g. 'bert', 'roberta', 'marianmt')
        model_dir (Path): the directory containing
        mar inputs.
        version (float, optional): the model version. Defaults to 1.0.
        serialized_file (str, optional): the model output of save_pretrained().
        Defaults to None.
        handler (str, optional): the serving handler file.
        Defaults to "text_handler.py".
    """
    export_dir = model_dir.parent.joinpath("mar")
    export_dir.mkdir(exist_ok=True)
    if model_type in ["bert", "roberta"]:
        files = EXTRA_FILES.copy()
    else:
        files = TRANS_FILES.copy()
    xfiles = ",./".join(shlex.quote(x) for x in files)
    if serialized_file is None:
        serialized_file = "pytorch_model.bin"
    if handler == "text_handler.py":
        copy_handler(model_dir)
    files.extend([serialized_file, handler])
    if mar_files(model_dir, files):
        # --model-file=./model.py
        cmd = f"""torch-model-archiver
            --model-name={model_dir.name}
            --version={version}
            --serialized-file=./{serialized_file}
            --handler=./{handler}
            --extra-files "./{xfiles}"
            --export-path={export_dir}
        """
        cm = subprocess.Popen(shlex.split(cmd), cwd=model_dir)
        cm.communicate()
        logger.info(f"torch archive {model_dir.name}.mar created")
