#!/usr/bin/env bash
BASE_DIR="/home/dcheswor/rdp-vscode-devcontainer/serve-quik"

# docker exec -it rdp-vscode-devcontainer_devcontainer_rdp-vscode-dev-build_1 python /workspaces/rdp-vscode-devcontainer/serve-quik/serve_quik/container/mar_switch.py
# docker stop local_test_project
# docker rm local_test_project
# docker image rm serve-test-project:latest
cd $BASE_DIR/serve_quik/container/
docker-compose --project-directory="${BASE_DIR}/deployments/test-project" up --detach
docker run -d -p 8180:8080 --name=local_test_project serve-test-project:latest

