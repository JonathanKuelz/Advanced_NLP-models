#!/bin/bash
# Forwards the visdom port of the remote server to localhost:7000

if [ -z "$1" ]
  then
    port=7000
else
  port="$1"
fi

echo "Enter Server Computer ID"
read ID
ssh -p 58022 -NL ${port}:localhost:${port} s0238@atcremers${ID}.informatik.tu-muenchen.de
