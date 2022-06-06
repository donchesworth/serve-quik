from pathlib import Path
import pytest
from serve_quik import arg
import pandas as pd
import json
from collections import OrderedDict
from os import getenv
import sys

# bd = Path("/workspaces/rdp-vscode-devcontainer/serve-quik")
# TESTDIR = bd.joinpath("serve_quik", "tests")
TESTDIR = Path(__file__).parent
SAMPLE = TESTDIR.joinpath("sample_data.json")
ENDPOINT_URL = getenv("ENDPOINT_URL", "https://localhost:5000")


@pytest.fixture
def args():
    """sample args namespace"""
    sys.argv = ["-p", "my_project", "-mt", "marianmt"]
    args = arg.parse_args(sys.argv)
    args.url = ENDPOINT_URL
    return args


@pytest.fixture(scope="session")
def senti_classes():
    """sentiment classes"""
    dir_classes = OrderedDict(
        [(0, "Negative"), (1, "Neutral"), (2, "Positive")]
    )
    return dir_classes


@pytest.fixture(scope="session")
def sample_data():
    """sample user/item dataset"""
    with open(SAMPLE) as f:
        df = pd.DataFrame(json.load(f))
    return df
