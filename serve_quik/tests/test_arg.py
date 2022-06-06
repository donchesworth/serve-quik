from serve_quik import arg
from argparse import Namespace
import sys


def test_args():
    sys.argv = ["-p", "my_project", "-mt", "marianmt"]
    args = arg.parse_args(sys.argv)
    assert isinstance(args, Namespace)
