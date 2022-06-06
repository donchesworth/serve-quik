from typing import List, Tuple, Optional, Dict
import json
import socket
from pathlib import Path
import logging
import sys
import hugging_quik as hq

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BASE_API_PORT = 8080
BASE_METRIC_PORT = 8082
PORT_SETS = list(zip(
    range(BASE_API_PORT, 10002, 100),
    range(BASE_METRIC_PORT, 10002, 100)
    ))


def file_path():
    return Path(__file__).resolve()


def set_serve_dir(project_name: str) -> Path:
    base_dir = file_path().parents[1]
    serve_dir = base_dir.joinpath("deployments", project_name)
    try:
        serve_dir.mkdir()
        logger.info(f"serve directory created in {serve_dir}")
    except FileExistsError as e:
        logger.info(
            f"{e.strerror}: the {project_name} project folder "
            "already exists. Some files will be overwritten."
            )
    return serve_dir


def set_model_dir(
    serve_dir: Path, model_type: str, kwargs: Dict[str, str]
) -> Path:
    model_name = hq.model.model_info(model_type, kwargs=kwargs)["model_name"]
    model_name = model_name.split("/")[-1]
    model_dir = serve_dir.joinpath(model_name)
    try:
        model_dir.mkdir()
        logger.info(f"model directory created in {model_dir}")
    except FileExistsError as e:
        logger.info(
            f"{e.strerror}: the {model_name} model folder "
            "already exists. Some files will be overwritten."
            )
    return model_dir


def txt_format(txt_arr: List[str]) -> str:
    """Format text to be predicted in a way that the serving API expects

    Args:
        txt_arr (List[str]): A list of texts

    Returns:
        str: A formatted string for the serving API
    """
    txt = f'{{"instances":' \
        f'{json.dumps([{"data": text} for text in txt_arr])}}}'
    return txt


def unused_port(port: int) -> bool:
    """Determine if the port is unused, by receiving a 111 error
    (connection refused).

    Args:
        port (int): The port to attempt connection

    Returns:
        bool: whether a port connection was refused (pointing to unused)
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    return s.connect_ex(('localhost', port)) == 111


def first_open_ports(
    port_sets: Optional[List[Tuple[int]]] = PORT_SETS
) -> Tuple[int]:
    """Provide the first set of open ports from a list of port sets.

    Args:
        port_sets (List[Tuple): A list of tuples which are possible ports
        to be used for an API.

    Returns:
        [type]: The first set of ports that are determined to be unused.
    """
    for ports in port_sets:
        if all([unused_port(port) for port in ports]):
            return ports
