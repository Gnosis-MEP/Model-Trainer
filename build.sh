#!/bin/bash
# https://www.docker.com/blog/how-to-rapidly-build-multi-architecture-images-with-buildx/
# docker buildx create --name mybuilde r--driver docker-container --use
# docker buildx create --name mybuilder --use
# docker buildx inspect --bootstrap
# update-binfmts --enable

# docker buildx build \
#     --platform linux/arm/v7 \
#     --pull \
#     --tag registry.insight-centre.org/sit/mps/felipe-phd/model-trainer:latest-rpi
#     # --push .



# docker buildx build --platform linux/arm/v7 --pull --tag registry.insight-centre.org/sit/mps/felipe-phd/model-trainer:latest-rpi -f Dockerfile.rpi --push .
# docker buildx build \
#     --platform linux/arm/v7 \
#     --pull \
#     --output type=docker \
#     --tag registry.insight-centre.org/sit/mps/felipe-phd/model-trainer:latest-rpi \
#     -f Dockerfile.rpi
#     --push .

# docker buildx build \
#     --platform linux/arm/v7 \
#     --pull \
#     --output type=docker \
#     --tag registry.insight-centre.org/sit/mps/felipe-phd/model-trainer:latest-rpi \
#     -f Dockerfile.rpi


docker buildx build \
    --pull \
    --load \
    --tag registry.insight-centre.org/sit/mps/felipe-phd/model-trainer:latest-rpi \
    --platform linux/arm/v7 \
    -f Dockerfile.rpi .


# run arm7 img in other arch:
# docker run --rm --privileged multiarch/qemu-user-static:register --reset
# docker run -it --rm --name my-container -v /usr/bin/qemu-arm-static:/usr/bin/qemu-arm-static registry.insight-centre.org/sit/mps/felipe-phd/model-trainer:latest-rpi