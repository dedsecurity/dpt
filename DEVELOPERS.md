# Installation

Git clone this repository, and `cd` into directory for remaining commands
```
git clone https://github.com/dedsecurity/gpt-ded && cd gpt-ded
```

## Native Installation

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

## Sample generation

To generate samples of the small model:
```
python src/generate_text.py
```

## Bee model (Chatbot)
```
python src/Bee.py
```
## Token model
```
python src/model_token.py
```
