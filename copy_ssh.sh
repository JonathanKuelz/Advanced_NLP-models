#!/bin/bash

echo "Enter Computer ID"
read ID
echo "Enter localpath"
read localpath


scp -r -P 58022 s0238@atcremers${ID}.informatik.tu-muenchen.de:/usr/prakt/s0238/pcss20-dnc/DNC/train_results ${localpath}

