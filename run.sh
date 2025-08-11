#! /bin/bash

export UVICORN_PORT=9988
cd app
granian --interface asgi main:app --host 0.0.0.0 --port $UVICORN_PORT
cd ..