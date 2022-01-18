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

