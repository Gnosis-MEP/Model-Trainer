# Model Trainer
A service that provide training for some types of models on the adaptive publisher


# Installation

## Configure .env
Copy the `example.env` file to `.env`, and inside it replace `SIT_PYPI_USER` and `SIT_PYPI_PASS` with the correct information.

## Installing Dependencies


### Using pip

Load the environment variables from `.env` file using `source load_env.sh`.

To install from the `requirements.txt` file, run the following command:
```
$ pip install  -r requirements.txt
```

# Running
Enter project python environment (virtualenv or conda environment)

**ps**: It's required to have the .env variables loaded into the shell so that the project can run properly. An easy way of doing this is using `pipenv shell` to start the python environment with the `.env` file loaded or using the `source load_env.sh` command inside your preferable python environment (eg: conda).

Then, run the service with:
```
$ ./model_trainer/run.py
```

#  Install PyTorch on Raspberry Pi
https://gist.github.com/wenig/8bab88dede5c838660dd05b8e5b2e23b
 sudo apt install libavcodec-dev libavformat-dev libswscale-dev

maybe this as well: https://pytorch.org/tutorials/intermediate/realtime_rpi.html

CUDA_VISIBLE_DEVICES="" python model_trainer/eval_cls_model_torch.py


# License
Distributed under the apache 2 license. See license file for more details.