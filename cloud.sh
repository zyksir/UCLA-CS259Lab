#!/bin/sh

# create you password
sudo passwd ec2-user
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install tmux
brew install llvm
brew install libomp
brew install cmake
brew install aria2
aria2c --header "Host: adcdownload.apple.com" --header "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8" --header "Upgrade-Insecure-Requests: 1" --header "Cookie: ADCDownloadAuth=#{token}" --header "User-Agent: Mozilla/5.0 (iPhone; CPU iPhone OS 10_1 like Mac OS X) AppleWebKit/602.2.14 (KHTML, like Gecko) Version/10.0 Mobile/14B72 Safari/602.1" --header "Accept-Language: en-us" -x 16 -s 16 #{url} -d ~/Downloads
xcode-select --install 
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer

# install anaconda: search link in https://www.anaconda.com/download
curl -o ~/anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2023.03-1-MacOSX-arm64.sh
bash ~/anaconda.sh
pip3 install torch torchvision torchaudio
pip3 install pytest-pytorch
conda env create --file environment.yaml
conda activate dlsys-needle-m1
python3 download_data.py
make
pip3 install -e .
