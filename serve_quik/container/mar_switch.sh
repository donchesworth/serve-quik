#!/usr/bin/env bash

docker exec -it rdp-vscode-devcontainer_devcontainer_rdp-vscode-dev-build_1 python /workspaces/rdp-vscode-devcontainer/nps-sent-serve/mar_switch.py
docker stop local_text_translate
docker rm local_text_translate
docker image rm serve-text-translate:latest
docker build --tag serve-text-translate:latest ./
docker run -d -p 8180:8080 --name=local_text_translate serve-text-translate:latest

