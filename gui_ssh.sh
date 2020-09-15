#!/bin/bash

echo "Enter Computer ID"
read ID
ssh -X -p 58022 s0238@atcremers${ID}.informatik.tu-muenchen.de
