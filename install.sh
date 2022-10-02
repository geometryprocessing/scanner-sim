#!/bin/bash
set -e

# bootstrap
sudo apt-get update
sudo apt-get install git python3-pip cmake
# clone mitsuba
if [ ! -d "mitsuba" ] ; then
    git clone https://github.com/skoch9/mitsuba
fi
# install mitsuba
cd mitsuba
cp build/config-linux-gcc.py config.py
sudo apt-get install build-essential scons mercurial qt4-dev-tools libpng-dev libjpeg-dev libilmbase-dev libxerces-c-dev libboost-all-dev libopenexr-dev libglewmx-dev libxxf86vm-dev libpcrecpp0v5 libeigen3-d$
scons -j 1 ; echo -e '\a'
# add to path via bashrc
echo "source $PWD/setpath.sh" >> ~/.bashrc
cd ..

# clone sls
if [ ! -d "scanner-sim" ] ; then
    git clone https://github.com/geometryprocessing/scanner-sim
fi
# install requirements
cat requirements.txt | xargs -t -i sh -c 'pip3 install -U {} || true'