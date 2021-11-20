# Installation

Git clone this repository, and `cd` into directory for remaining commands
```
git clone https://github.com/dedsecurity/gpt-ded && cd gpt-ded
```

## Installation

Install tensorflow
```
pip install tensorflow
```
or
```
pip install tensorflow-gpu
```

Install other python packages:
```
pip install -r requirements.txt
```

## Docker Installation

Build the Dockerfile and tag the created image as `gpt-ded`:
```
docker build --tag gpt-ded -f Dockerfile.gpu . # or Dockerfile.cpu
```

Start an interactive bash session from the `gpt-ded` docker image.

You can opt to use the `--runtime=nvidia` flag if you have access to a NVIDIA GPU
and a valid install of [nvidia-docker 2.0](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)).
```
docker run --runtime=nvidia -it gpt-ded bash
```

# Running

| WARNING: Samples are unfiltered and may contain offensive content. |
| --- |

Some of the examples below may include Unicode text characters. Set the environment variable:
```
export PYTHONIOENCODING=UTF-8
```
to override the standard stream settings in UTF-8 mode.

## Unconditional sample generation

To generate unconditional samples from the small model:
```
python3 src/generate_unconditional_samples.py | tee /tmp/samples
```
There are various flags for controlling the samples:
```
python3 src/generate_unconditional_samples.py --top_k 40 --temperature 0.7 | tee /tmp/samples
```

To check flag descriptions, use:
```
python3 src/generate_unconditional_samples.py -- --help
```

## Conditional sample generation

To give the model custom prompts, you can use:
```
python3 src/interactive_conditional_samples.py --top_k 40
```

To check flag descriptions, use:
```
python3 src/interactive_conditional_samples.py -- --help
```
