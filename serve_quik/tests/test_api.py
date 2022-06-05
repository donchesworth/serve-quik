import serve_quik as sq
import numpy as np
import pytest

BATCH_SIZE = 3
OUTPUT = '{"instances":[{"data": "this is absolutely terrible"}, {"data":'


def test_split_and_format(sample_data):
    txt_list = sq.api.split_and_format(
        sample_data.to_numpy()[:, 0],
        BATCH_SIZE
        )
    assert (txt_list[0][:63] == OUTPUT)


def test_batch_inference(sample_data, args):
    output = np.array([
        'Positive', 'Positive', 'Positive', 'Positive', 'Neutral',
        'Negative', 'Positive', 'Positive'], dtype=object)
    x = sample_data.to_numpy()[:, 0]
    sr = sq.api.ServeRequest(x, BATCH_SIZE, args.url)
    # only works  when there's a running service
    # rdf = sr.batch_inference()
    # assert np.array_equal(output, rdf['label'].values)
