#!/bin/sh

cd /opt/sq
# source ./.env
# pytest
pytest --cov=/opt/sq/serve_quik --cov-config=.coveragerc
# echo $(xmllint --xpath "string(//coverage/@line-rate)" coverage_cpu.xml)
# curl https://codecov.io/bash
