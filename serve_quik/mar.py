from pathlib import Path
from pytorch_quik import hugging, io, utils
from typing import List, Optional, OrderedDict, KeysView
from argparse import Namespace
import shlex
import subprocess
import requests
from urllib.parse import urlparse
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
    serve_config = hugging.model_info(model_type, labels, kwargs)
    io.json_write(serve_path, "setup_config.json", serve_config)


def save_index_to_name(serve_path: str, indexed_labels: OrderedDict[int, str]):
    """Create the required index_to_name file for serving

    Args:
        serve_path (str): The directory to store the sample.
        indexed_labels (OrderedDict[str, int]): the target labels with indexes.
    """
    io.json_write(serve_path, "index_to_name.json", indexed_labels)


def save_sample(serve_path):
    """A sample input to test the serving model

    Args:
        serve_path (str): The directory to store the sample.
    """
    sample = ["Great company with fast support"]
    sample = utils.txt_format(sample)
    io.json_write(serve_path, "sample_text.json", sample)


def save_handler(serve_path, url: Optional[str] = None):
    """Download the handler file for serving.

    Args:
        serve_path ([type]): the torch serve directory
        url (str, optional): The url where the handler can be found.
        Defaults to None.
    """
    if url is None:
        url = urlparse("https://raw.githubusercontent.com")
        handler_loc = Path(
            "donchesworth/pytorch-quik/main/",
            "pytorch_quik",
            "handler",
            "transformer_handler_pq.py",
        )
        url = url._replace(path=str(handler_loc))
    filename = Path(serve_path).joinpath(handler_loc.name)
    r = requests.get(url.geturl(), allow_redirects=True)
    open(filename, "wb").write(r.content)


def save_env_file(
    serve_path: Path,
    project_name: str,
    api_port: int,
    metric_port: int,
):
    """Create a .env file with args for docker-compose

    Args:
        serve_path (Path): the torch serve directory
        project_name (str): the project name as a base for the image
        and container names
        api_port (int): the host port to be paired with the serving 8080
        mertric_port (int): the host metric port to be paired with 8082

    """
    envs = {}
    envs["IMAGE_NAME"] = f'serve-{project_name}'
    envs["CONTAINER_NAME"] = f'local_{project_name.replace("-", "_")}'
    envs["API_PORT"] = api_port
    envs["METRIC_PORT"] = metric_port
    file = serve_path.joinpath(".env")
    with open(file, 'w') as f:
        for key, value in envs.items():
            f.write(f'{key}="{value}"\n')


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
        serve_path = io.id_str("state_dict", args).parent.joinpath("serve")
    serve_path.mkdir(parents=True, exist_ok=True)
    save_setup_config(serve_path, indexed_labels.keys(), args)
    save_index_to_name(serve_path, indexed_labels)
    save_sample(serve_path)
    save_handler(serve_path)
    save_env_file(serve_path, args)


def create_mar(
    args: Namespace,
    model_dir: Optional[Path] = None,
    version: Optional[float] = 1.0,
    serialized_file: Optional[str] = None,
    handler: Optional[str] = "text_handler.py",
):
    """build a torch-model-archive file using
    https://github.com/pytorch/serve/tree/master/model-archiver

    Args:
        args (Namespace): the project argparse namespace.
        model_dir (Path, optional): the directory containing
        mar inputs, and the output directory. Defaults to None.
        model_name (str, optional): the name of the model to be served.
        Defaults to None.
        version (float, optional): the model version. Defaults to 1.0.
        serialized_file (str, optional): the model output of save_pretrained().
        Defaults to None.
        handler (str, optional): the serving handler file.
        Defaults to "text_handler.py".
    """
    if model_dir is None:
        model_dir = Path(io.id_str("", args)).parent.joinpath("serve")
    export_dir = model_dir.joinpath("mar")
    export_dir.mkdir(parents=True, exist_ok=True)
    if args.model_type in ["bert", "roberta"]:
        files = EXTRA_FILES
    else:
        files = TRANS_FILES
    xfiles = ",./".join(shlex.quote(x) for x in files)
    model_id = args.model_name.split('-')
    model_id.extend(args.kwargs.values())
    model_id = '_'.join(model_id)
    if serialized_file is None:
        serialized_file = "pytorch_model.bin"
    handler = f"../../{handler}"
    files.extend([serialized_file, handler])
    if mar_files(model_dir, files):
        # --model-file=./model.py
        cmd = f"""torch-model-archiver
            --model-name={model_id}
            --version={version}
            --serialized-file=./{serialized_file}
            --handler=./{handler}
            --extra-files "./{xfiles}"
            --export-path={export_dir}
        """
        cm = subprocess.Popen(shlex.split(cmd), cwd=model_dir)
        cm.communicate()
        logger.info(f"torch archive {model_id}.mar created")


def start_container(serve_path):
    cmd = f"""docker-compose
        --project-directory="{serve_path.name}"
        up
        --detach
    """
    sc = subprocess.Popen(shlex.split(cmd), cwd=serve_path.parent)
    sc.communicate()

    logger.info(f"torch serve {serve_path.name} container started")
