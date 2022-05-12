#!/bin/sh

cd /opt/sq
source ./.env
# pytest
pytest --mpl --cov=/opt/sq/pytorch_quik --cov-config=.coveragerc --cov-report=xml:coverage_cpu.xml
# --ignore=pytorch_quik/tests/test_mlflow.py
echo $(xmllint --xpath "string(//coverage/@line-rate)" coverage_cpu.xml)
# curl https://codecov.io/bash
