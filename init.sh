sudo apt-get -y install tmux

git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm

cp ./conf/.tmux.conf ~/
cp ./conf/.tmux.conf.local ~/
cp ./conf/.tmux.conf.local ~/

tmux source-file ~/.tmux.conf

cp /blob/v-haoheliu/jobs_env/conf/.vimrc ~/
cp /blob/v-haoheliu/jobs_env/conf/.bashrc ~/

source ~/.bashrc
source ~/.vimrc

sudo apt-get -y install vim 

curl -k -L http://hengyunabc.github.io/bash_completion_install.sh | sh
bind -f ~/.inputrc

pip3 install phonemizer
sudo apt -y install fish
pip3 uninstall -y torch
pip install torch==1.6.0
pip install ipdb
sudo apt -y install psmisc

echo "start ssh-key generation"
# ssh-keygen -t rsa
ssh-keygen -f id_rsa -t rsa -N ''
cat ~/.ssh/id_rsa.pub
git config --global user.name "haoheliu"
git config --global user.email "867390095@qq.com"
pip3 install progressbar
sudo apt -y install espeak

cp ./conf/authorized_keys ~/.ssh/

# echo "Adding swap file"

# sudo swapon --show
# free -h
# df -h
# echo "allocating"
# sudo dd if=/dev/zero of=~/swapfile count=1024 bs=1MiB
# # sudo fallocate -l 32G /swapfile
# echo "done"
# ls -lh ~/swapfile
# sudo chmod 600 ~/swapfile
# ls -lh ~/swapfile
# sudo mkswap -f ~/swapfile
# echo "swap on!"
# sudo swapon ~/swapfile
# sudo swapon --show
# free -h
# echo "done"
