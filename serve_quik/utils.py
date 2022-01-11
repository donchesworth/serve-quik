from typing import List
import json


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
