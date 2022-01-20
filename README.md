# serve-quik

For creating torch archived transformer models and TorchServe containers (much quik-er)

## Summary

The process of building a torch model archive, building a torch serve container, determining the appropriate ports, and testing your container can be tedious, so I tried to create an automated process for all of this. Most of my automation is centered around MarianMT models, but can be used for other models (I use it for a BERT model). For instance, to build and deploy a container with some MarianMT models that will translate from Japanese, German, and Spanish to English, you could run the following:

``` bash
python main.py -p "text-translate" -mt marianmt -src ja de es
```

To test this, you could then run:

``` python
>>> import pytorch_quik as pq
>>> import pandas as pd
>>> import numpy as np

>>> text_dict = {
...     "opus-mt-ja-en": ["口は災いの元"],
...     "opus-mt-de-en": ["Alles hat ein Ende, nur die Wurst hat zwei"],
...     "opus-mt-es-en": ["Es tan corto el amor y tan largo el olvido"]
... }
>>> res = pd.DataFrame()
>>> for key, value in text_dict.items():
...     x = np.array(value, dtype='object')
...     url = f"http://localhost:8180/predictions/{key}"
...     sr = pq.api.ServeRequest(x, 2, url)
...     df = sr.batch_inference()
...     res = pd.concat([res, df])
... 
INFO:pytorch_quik.api:Batch 0, status_code: 200
INFO:pytorch_quik.api:Batch 0, status_code: 200
>>> print(res)
                                       translation
0                            The mouth is a curse.              
0       Love is so short and forgetfulness so long
0  Everything has an end, only the sausage has two
```

Pretty cool right? But what exactly is being automated? Hypothetically, any `huggingface.co` tokenizer and model could be placed into a torch archive and served with TorchServe. serve-quik does the following for this to happen: 

### Pull and prepare tokenizer

I've only implemented BERT, RoBERTA, and MarianMT, but more are to come. The tokenizer functions do the following:
- maps a string to a model name and tokenizer, such as:
    - `bert` to `bert-base-uncased` and `BertTokenizer`
    - `roberta` to `roberta-base` and `RobertaTokenizer`
    - `marianmt` _(with source and target like es and en)_ to `Helsinki-NLP/opus-mt-es-en` and `MarianTokenizer`
- pulls the appropriate tokenizer, then converts the cached tokenizer files to the input files `config.json`, `tokenizer_config.json`, `special_tokens_map.json`, then does the same for the tokenizer specific files, such as :
    - `index_to_name.json`, `sample_text.txt`, `vocab.txt` for sequence_classification models
    - `vocab.json`, `source.spm` and `target.spm` for sequence_to_sequence models

### Pull and prepare model

To prepare a model, you pull it, add model weights, and then save it. If you are using the pretrained model as-is, you can just provide the weights already in the model. The steps are:
- mapping a string to a model name, such as:
    - `bert` to `BertForSequenceClassification`
    - `roberta` to `RobertaForSequenceClassification`
    - `marianmt` to `MarianMTModel`
- pulling the pretrained model
- builds the model archive's `setup_config.json` file with defaults

{% note %}
**Note:** If you aren't providing your own trained weights (`state_dict`), you can just provide back the original weights to the pulled model `model.state_dict()`
{% endnote %}

### Dockerfile automation

Usually a container is built with a Dockerfile, docker-compose, or both. Although most TorchServe API containers are similar, there will always be differences, such as port numbers and container name. serve-quik takes these steps:

- _Determine ports_: Search for ports similar to the container's 8080 for the Inference API and 8082 for the Metrics API that aren't being used (in a **80 and **82 pattern)
- _Build `.env` file_: In order to use a common Dockerfile and docker-compose, a `.env` is built with `CONTAINER_DIR`, `IMAGE_NAME`, `CONTAINER_NAME`, `DIR_NAME`, `API_PORT`, and `METRIC_PORT`.
- _Build and start container_: Using the model archive directory, docker-compose directory, and `.env` file, build a torchserve container, and start it on the determined ports. the basic process is to `cd` to the `serve_quik/container` directory, 


