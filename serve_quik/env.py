from . import utils
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class EnvVars:
    """dot env variables for container building"""
    IMAGE_NAME: str
    CONTAINER_NAME: str
    API_PORT: int
    METRIC_PORT: int


def build_dot_env(serve_dir: Path):
    """Using api_name and available ports, build the .env file
    to be used by docker-compose to build the container

    Args:
        project_name (str): name of the api with dashes, e.g. nps-sentiment
    """
    apip, metp = utils.first_open_ports()
    envvars = EnvVars(
        IMAGE_NAME=f"serve-{serve_dir.name}",
        CONTAINER_NAME=f"local_{serve_dir.name.replace('-', '_')}",
        API_PORT=apip,
        METRIC_PORT=metp,
    )

    env = serve_dir.joinpath(".env")
    with open(env, 'w') as env_file:
        for key, value in asdict(envvars).items():
            env_file.write('%s=%s\n' % (key, value))
