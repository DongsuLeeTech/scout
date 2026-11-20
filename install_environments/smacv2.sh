#!/bin/bash

export SC2PATH=/data/StarCraftII

echo 'Adding SMAC maps.'
MAP_DIR="$SC2PATH/Maps/"
echo 'MAP_DIR is set to '$MAP_DIR
mkdir -p $MAP_DIR

wget https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip
unzip SMAC_Maps.zip -d SMAC_Maps
mv SMAC_Maps $MAP_DIR
rm -rf SMAC_Maps.zip

echo 'StarCraft II is installed.'
