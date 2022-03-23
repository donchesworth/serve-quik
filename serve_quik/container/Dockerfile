FROM pytorch/torchserve:0.5.3-cpu

USER root
RUN echo '\nservice_envelope=json' >> /home/model-server/config.properties && \
    pip install transformers[sentencepiece]
USER model-server

WORKDIR /home/model-server
RUN mkdir /home/model-server/model_store
COPY ./mar/*.mar /home/model-server/model_store/

CMD ["torchserve", \
     "--start", \
     "--model-store model_store", \
     "--models all", \
     "--ncs"]
