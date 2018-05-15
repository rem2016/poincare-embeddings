git clone https://github.com/rem2016/poincare-embeddings
cd poincare-embeddings
git checkout dev
sudo apt -y install tree
sudo apt -y install glances
bash bash/env/env.sh
python bash/env/download_wornet.py
sudo apt update
sudo apt upgrade
