import serve_quik as sq
import multiprocessing as mp
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from requests import Session
from typing import List
from multiprocessing.connection import Connection
import pandas as pd
import numpy as np
from math import ceil
import logging
import sys

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

JSON_HEADER = {"Content-Type": "application/json"}
#  sample_text = '{"instances":[{"data": "me encanta Red Hat"}]}'


def split_and_format(arr: np.array, length: int) -> List[str]:
    """Taking a numpy array of text, split into batches for separate API
    posts, and format into the torch serve required "instances" and "data."

    Args:
        arr (np.array): An array of text to be predicted
        length (int): The length of the batch, or batch size

    Returns:
        List[str]: A list of strings formatted for a Transformer handler.
    """
    splits = ceil(len(arr) / length)
    arr_list = np.array_split(arr.flatten(), splits)
    data_list = [sq.utils.txt_format(arr) for arr in arr_list]
    return data_list


def request_post(
    url: str, data_batch: str, sesh: Session, conn: Connection, num: int
):
    """send a POST request based on a requests_session and a connection

    Args:
        data_batch (str): A batch of data to be predicted
        sesh (Session): A session from requests_session
        conn (Connection): The input_pipe from mp.Pipe
        num (int): the batch number for when the data is recombined.
    """
    r = sesh.post(url, data=bytes(data_batch, "utf-8"), headers=JSON_HEADER)
    logger.info(f"Batch {num}, status_code: {r.status_code}")
    conn.send(r)


class ServeRequest:
    """A class for sending the request and receiving the response from
    the torch serve service API"""

    def __init__(
        self,
        text: np.array,
        batch_size: int,
        url: str,
    ):
        """The ServeRequest constructor

        Args:
            text (np.array): The set of text (or survey responses) to
            be predicted
            batch_size (int): The size of each batch request
            url (str): the url endpoint of the api

        """
        self.url = url
        self.data_list = split_and_format(text, batch_size)
        self.sesh = self.requests_session()

    def requests_session(self) -> Session:
        """Create an API session that can queue and recieve multiple
        requests. It can also retry when a request returns a 507 instead
        of a 200.

        Returns:
            Session: A requests session
        """
        retry_strategy = Retry(
            total=10,
            backoff_factor=1,
            status_forcelist=[507, 500],
            method_whitelist=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        sesh = Session()
        sesh.mount("http://", adapter)
        return sesh

    def batch_inference(self) -> pd.DataFrame:
        """Take an array of text fields, and return predictions via API

        Returns:
            pd.DataFrame: A dataframe with the original text, logits, and
            predicted label.
        """
        processes = []
        r_list = []
        preds = []
        self._raw_output = []
        for num, batch in enumerate(self.data_list):
            output_pipe, input_pipe = mp.Pipe(duplex=False)
            proc = mp.Process(
                target=request_post,
                args=(self.url, batch, self.sesh, input_pipe, num),
            )
            processes.append(proc)
            r_list.append(output_pipe)
            proc.start()
        [proc.join() for proc in processes]
        for r in r_list:
            pred = r.recv().json()["predictions"]
            self._raw_output.append(pred)
            # translation
            if not isinstance(pred[0], dict):
                preds.extend(pred[0])
            # sentiment
            else:
                preds.append(pd.json_normalize(pred, sep="_"))
        # translation
        if not isinstance(preds[0], pd.DataFrame):
            return pd.DataFrame(preds, columns=["translation"])
        # sentiment
        else:
            return pd.concat(preds, ignore_index=True)
