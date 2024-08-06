#! /usr/bin/env bash

source ../venv/bin/activate
python setup.py build_ext
cp build/lib*/_encoder/* .
