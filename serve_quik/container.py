from . import utils
from dataclasses import dataclass, asdict
from pathlib import Path
import shutil
import shlex
import subprocess
import logging
import sys

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CONTAINER_DIR = Path(__file__).parent.joinpath("container").resolve()


@dataclass
class EnvVars:
    """dot env variables for container building"""
    CONTAINER_DIR: Path
    IMAGE_NAME: str
    CONTAINER_NAME: str
    DIR_NAME: str
    API_PORT: int
    METRIC_PORT: int


def build_dot_env(serve_dir: Path):
    """Using api_name and available ports, build the .env file
    to be used by docker-compose to build the container

    Args:
        project_name (str): name of the api with dashes, e.g. nps-sentiment
    """
    api_port, met_port = utils.first_open_ports()
    project = serve_dir.name
    envvars = EnvVars(
        CONTAINER_DIR=CONTAINER_DIR,
        IMAGE_NAME=f"serve-{project}",
        CONTAINER_NAME=f"local_{project.replace('-', '_')}",
        DIR_NAME=project,
        API_PORT=api_port,
        METRIC_PORT=met_port,
    )
    env = serve_dir.joinpath(".env")
    with open(env, 'w') as env_file:
        for key, value in asdict(envvars).items():
            env_file.write(f'{key}="{value}"\n')


def start_container(serve_dir):
    logger.info(f"building a container with project-directory {serve_dir}")
    logger.info(f"running command in the directory {CONTAINER_DIR}")
    
    shutil.rmtree(serve_dir.joinpath("tmp"), ignore_errors=True)
    cmd = f"""docker-compose
        --project-directory="{serve_dir}"
        up
        --detach
    """
    sc = subprocess.Popen(shlex.split(cmd), cwd=CONTAINER_DIR)
    sc.communicate()

    logger.info(f"torch serve {serve_dir.name} container started")
