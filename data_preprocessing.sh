#!/bin/bash

## Five top incident roads
### 1210005301 1030001902 1220005401 1210003000 1130052300

## data preprocessing method
### mtsc / cls / SAD / mprofile / mprofile2

for sid in 1210005301 1030001902 1220005401 1210003000 1130052300
do
    echo $sid
    python 3.\ data_preprocessing_mprofile.py --target-sid $sid
done
