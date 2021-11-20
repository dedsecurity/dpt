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
