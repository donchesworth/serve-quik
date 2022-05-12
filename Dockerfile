FROM quay.io/donchesworth/rapids-dask-pytorch:py38-cuda10.2-rapids21.10-pytorch1.9-ubi8

# Labels
LABEL maintainer="Don Chesworth<donald.chesworth@gmail.com>"
LABEL org.label-schema.schema-version="0.0.0"
LABEL org.label-schema.name="serve-quik-test"
LABEL org.label-schema.description="functions to make working with TorchServe quik-er"

RUN pip install matplotlib sklearn

# Project installs
WORKDIR /opt/sq
COPY ./ /opt/sq/
RUN pip install .

RUN chgrp -R 0 /opt/sq/ && \
    chmod -R g+rwX /opt/sq/ && \
    chmod +x /opt/sq/entrypoint.sh

ENTRYPOINT ["/opt/sq/entrypoint.sh"]
