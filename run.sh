#! /bin/bash

export UVICORN_PORT=9988
cd app
uvicorn main:app --host 0.0.0.0