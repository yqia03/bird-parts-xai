# Optional frozen-model inference

This is the original single-input adapter for **CaBRNet 1.2's PIPNet / ConvNeXt
Tiny / CUB-200 / seed 0 / final** checkpoint. It loads the published trained
parameters strictly, freezes the model, and exports raw target-class evidence
and contributions from all prototype channels. It applies no original-PIPNet
`inference=True` / 0.1 threshold and performs no training.

The separate [analysis example](../analysis/README.md) works with Python's
standard library. Running this optional entry requires a model environment and
the approximately 112 MB checkpoint. Weights and image inputs are not included.

## Install the original inference environment

Commands below run from this repository's root. Use **Python 3.12** for the
original Mac CPU combination. This preserves the project's inference dependency
snapshot; it is not an upstream training/GUI/CLI environment or a cross-platform
lock file. CaBRNet eagerly imports attribution, plotting and ONNX modules even
for this forward path, so the practical inference dependencies include those
packages. Torch and torchvision are installed as a paired version, and CaBRNet
is installed separately with `--no-deps`.

```sh
python3.12 -m venv inference/.venv
inference/.venv/bin/python -m pip install torch==2.6.0 torchvision==0.21.0
inference/.venv/bin/python -m pip install -r inference/requirements-inference.txt
inference/.venv/bin/python -m pip install cabrnet==1.2 --no-deps
```

## Obtain the checkpoint

Published model source: [CaBRNet weight release on Zenodo](https://zenodo.org/records/18429182).
The original downloader uses HTTP byte ranges to retrieve the chosen final model
and companion files instead of the complete 2.477 GB archive:

```sh
python3 inference/scripts/download_weights.py
```

It requires server support for byte ranges and checks each extracted member's
ZIP CRC. It records a manifest with the received files' SHA-256 digests; it
does not check the entire archive's MD5 because it retrieves only selected
members. If byte ranges are unavailable, download the release manually and
extract `pipnet_cub200_convnext_s0/final/` into
`inference/weights/cabrnet_cub_s0_final/`. Keep the matching `model_arch.yml`
and `model_state.pth` together. This is a CaBRNet-trained checkpoint, distinct
from the original PIP-Net authors' weights.

Expected `model_state.pth` SHA-256:

```text
db5c72aa4fc29152d61471567f5b6ee0ddf3eb273c16090968002c7a1a060eb7
```

## Run a synthetic CPU check

```sh
inference/.venv/bin/python inference/scripts/quickstart.py --device cpu --target 0 --output inference/runs/synthetic
```

Without `--image`, this uses a seeded synthetic tensor to check loading and
forward computation. It checks the 200-class output shape, repeated-forward
agreement, unchanged model state, and reconstruction of all class scores from
prototype presence × nonnegative classifier weights. It writes
`verification.json` and `prototype_contributions.csv`. Synthetic input does
not measure bird recognition accuracy.

For an image you have permission to use, supply its path and a fixed zero-based
CUB class index. The optional `--classes` argument accepts the official CUB
`classes.txt` file with all 200 class IDs:

```sh
inference/.venv/bin/python inference/scripts/quickstart.py --device cpu --image YOUR_IMAGE.jpg --target 0 --output inference/runs/image
```

Image preprocessing is RGB conversion, resize to 224 × 224, tensor conversion,
and ImageNet channel normalization. Scores are native nonnegative evidence,
not probabilities. A prototype's peak grid location is not its full receptive
field. `--model-dir` accepts an alternative local checkpoint directory; default
weights and outputs are resolved relative to the `inference/` folder. Explicit
relative argument paths are resolved from the working directory.

## Verification and provenance

Both scripts and the dependency snapshot were copied unchanged from the
original project adapter; exact hashes are recorded in `provenance.json`.
Both `--help` commands were rerun successfully when preparing this public
entry. Inference was not rerun in that preparation environment because its
available Python runtimes lacked torch, torchvision and CaBRNet. No packages
were installed and no network download was performed. The original project
documented Python 3.12 / torch 2.6.0 / torchvision 0.21.0 / CaBRNet 1.2 on Mac
CPU; this public extraction does not add a fresh hardware validation claim.

Weights, downloaded metadata, inputs, output reports and the virtual environment
remain local through this folder's `.gitignore`.
