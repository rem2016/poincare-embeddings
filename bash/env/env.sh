conda install -y pytorch-cpu torchvision-cpu -c pytorch
git submodule init
git submodule update
pip install spacy
pip install sparqlwrapper
pip install rdflib
pip install gensim
python bash/env/download_wordnet.py

