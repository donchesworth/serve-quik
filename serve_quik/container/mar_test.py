import pytorch_quik as pq
import pandas as pd
import numpy as np

text_dict = {
    "opus-mt-es-en": [
        "Conocer el amor de los que amamos es el fuego que alimenta la vida",
        "Podrán cortar todas las flores, pero no podrán detener la primavera",
        "Hay un cierto placer en la locura, que solo el loco conoce",
        "Es tan corto el amor y tan largo el olvido",
        ],
    "opus-mt-de-en": [
        "Übung macht den Meister",
        "Bald reif hält nicht steif",
        "Nur die Harten kommen in den Garten",
        "Alles hat ein Ende, nur die Wurst hat zwei"
    ]
}

res = pd.DataFrame()

for key, value in text_dict.items():
    x = np.array(value, dtype='object')
    url = f"http://deepshadow.gsslab.rdu2.redhat.com:8180/predictions/{key}"
    sr = pq.api.ServeRequest(x, 2, url)
    df = sr.batch_inference()
    pd.concat([res, df])

