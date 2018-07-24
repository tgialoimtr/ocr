Use PYTHON 2.7, UBUNTU 16.04

### download opencv-3.2 and opencv_contrib-3.2:

    mkdir ~/Downloads
    cd ~/Downloads
    wget -O opencv.zip https://github.com/opencv/opencv/archive/3.2.0.zip
    unzip opencv.zip
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.2.0.zip
    unzip opencv_contrib.zip
### Install dependencies for opencv

    sudo apt-get update
    sudo apt-get install build-essential cmake pkg-config
    sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
    sudo apt-get install libgtk-3-dev
    sudo apt-get install libatlas-base-dev gfortran
    sudo apt-get install python2.7-dev python3.5-dev

### Build and install opencv

    cd ~/Downloads/opencv-3.2.0/
    mkdir build
    cd build
    pip install numpy
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D OPENCV_EXTRA_MODULES_PATH=~/Downloads/opencv_contrib-3.2.0/modules/ -D WITH_CUDA=OFF -D ENABLE_PRECOMPILED_HEADERS=OFF ..
(Verify features to build: text, features_2d)

(If build for python3, if build for java ?)

    make -j4
    sudo make install
    sudo ldconfig
    ln lib/cv2.so <PYTHON2 SITE PACKAGE>/cv2.so
(Verify cv2: import cv2)

### Install tesseract 4.0 dependencies:

    sudo apt-get install autoconf autoconf-archive automake libtool
    sudo apt-get install pkg-config
    sudo apt-get install libpng-dev
    sudo apt-get install libjpeg8-dev
    sudo apt-get install libtiff5-dev
    sudo apt-get install zlib1g-dev

### Install leptonica 1.74:

    cd ~/Downloads
    wget http://www.leptonica.org/source/leptonica-1.74.4.tar.gz
    tar xvzf leptonica-1.74.4.tar.gz
    cd leptonica-1.74.4/
    ./configure
    make
    sudo make install
    sudo ldconfig

### Install tesseract 4.0:

    cd ~/Downloads
    wget https://github.com/tesseract-ocr/tesseract/archive/4.0.0-beta.3.tar.gz
    tar xvzf 4.0.0-beta.3.tar.gz
    cd tesseract-4.0.0-beta.3/
    ./autogen.sh
    ./configure
    make
    sudo make install

### Download tesseract Vietnamese and English data:

    wget https://github.com/tesseract-ocr/tessdata/raw/fb1266d52b0a93ef27dbff54ecd422809c9c4f68/vie.traineddata
    sudo mv vie.traineddata /usr/local/share/tessdata/
    wget https://github.com/tesseract-ocr/tessdata/raw/fb1266d52b0a93ef27dbff54ecd422809c9c4f68/eng.traineddata
    sudo mv eng.traineddata /usr/local/share/tessdata/
 (Verify tesseract)

### Install Ocropus library:

    cd ~/Downloads
    wget -O ocropus.zip https://github.com/tmbdev/ocropy/archive/v1.3.3.zip
    unzip ocropus.zip
    cd ocropy-1.3.3/
    pip install -r requirements.txt
    python setup.py install

### Install Aspell:

    sudo apt-get install aspell
    cd ~/Downloads
    wget https://ftp.gnu.org/gnu/aspell/dict/vi/aspell6-vi-0.01.1-1.tar.bz2
    tar xvjf aspell6-vi-0.01.1-1.tar.bz2
    cd aspell6-vi-0.01.1-1/
    ./configure
    make
    sudo make install

### Install App's dependencies and and configuration:

    sudo apt-get install enchant
    pip install pyenchant
    pip install Flask scipy scikit-image bs4 pytesseract==0.1.7 editdistance
    mkdir ~/workspace
    cd ~/workspace
    git clone https://github.com/tgialoimtr/ocr.git
    cd ocr/src
    vi utils/common.py
Edit TEMPORARY_PATH to absolute path of "ocr/temp", which should be ~/workspace/ocr/temp

(Configure args.logsdir and args.download_dir, or just make new these folders:)
    mkdir /tmp/temp /tmp/logs

### Run App:

    cd ~/workspace/ocr/src
    PYTHONPATH=`pwd` python controller/webapp.py

Access  < server-address > :8080/ocr/cmnd9

Upload image of CMND (new, old, ...) to demo

