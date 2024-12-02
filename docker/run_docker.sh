xhost si:localuser:root

docker run -it \
           --privileged \
           --name quad_docker \
           --net=host \
           --env="DISPLAY" \
	   --gpus all \
           --env="QT_X11_NO_MITSHM=1" \
           quad_docker \
           bash

