#!/bin/bash

path="/mnt/md0/new-home/kent1201/文件/6Dof/" 
image="ccucsmvllab/5gimage"

docker run -it --runtime=nvidia -v ./6Dof/Dockershare:/6Dof/Dockershare ccucsmvllab/5gimage 
