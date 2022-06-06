import serve_quik as sq
import pandas as pd
import numpy as np

text_dict = {
    "opus-mt-ja-en": [
        "口は災いの元",
        "能ある鷹は爪を隠す",
        "猿も木から落ちる",
        "悪銭身に付かず",
    ],
    "opus-mt-de-en": [
        "Übung macht den Meister",
        "Bald reif hält nicht steif",
        "Nur die Harten kommen in den Garten",
        "Alles hat ein Ende, nur die Wurst hat zwei",
    ],
    "opus-mt-es-en": [
        "Conocer el amor de los que amamos es el fuego que alimenta la vida",
        "Podrán cortar todas las flores, pero no podrán detener la primavera",
        "Hay un cierto placer en la locura, que solo el loco conoce",
        "Es tan corto el amor y tan largo el olvido",
    ],
}

res = pd.DataFrame()

for key, value in text_dict.items():
    x = np.array(value, dtype='object')
    url = f"http://deepshadow.gsslab.rdu2.redhat.com:8180/predictions/{key}"
    sr = sq.api.ServeRequest(x, 2, url)
    df = sr.batch_inference()
    res = pd.concat([res, df])

