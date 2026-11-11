#!/bin/sh
# SIFT1M from the INRIA TEXMEX corpus (http://corpus-texmex.irisa.fr/), ~160 MB download. Not redistributed here.
set -e
mkdir -p data && cd data
curl -o sift.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz && rm sift.tar.gz
