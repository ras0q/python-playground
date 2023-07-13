#!/bin/sh

DIR=$1
PYTHON_VERSION=$2

mkdir -p $DIR

echo "python $PYTHON_VERSION" > $DIR/.tool-versions

echo "use asdf" > $DIR/.envrc
echo "source .venv/bin/activate" >> $DIR/.envrc
direnv allow $DIR

cd $DIR
python3 -m venv .venv
