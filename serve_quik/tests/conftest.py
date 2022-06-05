from pathlib import Path
import pytest
from serve_quik import arg
import pandas as pd
import json
from collections import OrderedDict
import sys

# bd = Path("/workspaces/rdp-vscode-devcontainer/pytorch-quik")
# TESTDIR = bd.joinpath("pytorch_quik", "tests")
TESTDIR = Path(__file__).parent
SAMPLE = TESTDIR.joinpath("sample_data.json")


@pytest.fixture
def args():
    """sample args namespace"""
    sys.argv = ["-p", "my_project"]
    args = arg.parse_args(sys.argv)
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


