"""Frozen CaBRNet 1.2 PIPNet inference, with auditable native contributions.

Without --image this performs a seeded synthetic-input loading check only.
For an actual study, supply --image, --target and the CUB classes.txt file.
Native CaBRNet forward has NO original-PIPNet inference=True / 0.1 clamp.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import tempfile
import time


REPO_ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def model_hash(model):
    h = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_model(model_dir, device):
    import torch
    import yaml
    from cabrnet.archs.generic.model import CaBRNet
    cfg = yaml.safe_load((Path(model_dir)/'model_arch.yml').read_text())
    if cfg['classifier']['name'] != 'PIPNetClassifier' or cfg['classifier']['params']['num_classes'] != 200:
        raise ValueError('This adapter requires the CUB200 PIPNet checkpoint')
    # The complete trained extractor is in model_state.pth. Avoid an unnecessary
    # download of the ImageNet initialization mentioned in the published YAML.
    cfg['extractor']['backbone']['weights'] = ''
    model = CaBRNet.build_from_config(cfg)
    state = torch.load(Path(model_dir)/'model_state.pth', map_location='cpu', weights_only=True)
    # Use the standard strict loader to obtain and check the incompatibility report.
    incompatible = torch.nn.Module.load_state_dict(model, state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=REPO_ROOT / 'weights' / 'cabrnet_cub_s0_final')
    parser.add_argument('--image', type=Path)
    parser.add_argument('--target', type=int, help='Fixed zero-based CUB class index')
    parser.add_argument('--classes', type=Path, help='Official CUB classes.txt; optional for loading check')
    parser.add_argument('--device', choices=['cpu','cuda','mps','auto'], default='cpu')
    parser.add_argument('--output', type=Path, default=REPO_ROOT / 'runs' / 'cabrnet_single')
    args = parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'cs5495-pipnet-mpl'))
    import torch
    import torchvision
    from PIL import Image
    from torchvision import transforms
    if args.target is not None and not 0 <= args.target < 200:
        parser.error('--target must be in [0,199]')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    start = time.perf_counter()
    model = load_model(args.model_dir,device)
    load_seconds = time.perf_counter()-start
    before = model_hash(model)
    if args.image:
        transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                      transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        x=transform(Image.open(args.image).convert('RGB')).unsqueeze(0).to(device)
        input_kind='user_supplied_image'
    else:
        generator=torch.Generator().manual_seed(5495)
        x=torch.rand((1,3,224,224),generator=generator).to(device)
        input_kind='synthetic_loading_check_not_research_data'
    def synchronize():
        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()

    with torch.inference_mode():
        synchronize()
        start=time.perf_counter()
        maps,presence,scores=model(x)
        synchronize()
        forward_seconds=time.perf_counter()-start
        _,_,again=model(x)
    if scores.shape != (1,200) or not torch.isfinite(scores).all():
        raise RuntimeError('Unexpected model output shape or non-finite scores')
    torch.testing.assert_close(scores,again,atol=1e-6,rtol=1e-5)
    layer=model.classifier.last_layer
    weights=layer.weight.detach().relu()
    reconstructed=presence @ weights.T
    if layer.bias is not None:
        reconstructed=reconstructed+layer.bias
    torch.testing.assert_close(reconstructed,scores,atol=1e-5,rtol=1e-4)
    after=model_hash(model)
    if before != after:
        raise RuntimeError('Model state changed during inference')
    target=args.target if args.target is not None else int(scores[0].argmax())
    no_evidence=bool((scores[0] == 0).all())
    class_names={}
    if args.classes:
        for line in args.classes.read_text().splitlines():
            key,value=line.strip().split(maxsplit=1)
            index = int(key)-1
            if index in class_names or not re.match(rf'^{index+1:03d}\.', value):
                raise ValueError('Expected unique official CUB class IDs with matching numeric directory prefixes')
            class_names[index]=value
        if set(class_names) != set(range(200)):
            raise ValueError('classes.txt must contain all 200 official CUB classes (IDs 1..200)')
    contribution=presence[0]*weights[target]
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'prototype_contributions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['prototype_id','native_presence','target_weight','target_contribution','peak_grid_row','peak_grid_col'])
        for p in range(presence.shape[1]):
            peak=int(maps[0,p].argmax())
            writer.writerow([p,float(presence[0,p]),float(weights[target,p]),float(contribution[p]),peak//maps.shape[-1],peak%maps.shape[-1]])
    report={
        'model':'CaBRNet-trained PIPNet CUB200 ConvNeXt seed0 final',
        'original_pipnet_checkpoint':False,'training_performed':False,
        'input_kind':input_kind,'image':str(args.image) if args.image else None,
        'image_sha256':file_sha256(args.image) if args.image else None,
        'classes_sha256':file_sha256(args.classes) if args.classes else None,
        'preprocessing':'RGB; resize 224x224; ToTensor; ImageNet channel normalization',
        'device':device,'versions':{'torch':torch.__version__,'torchvision':torchvision.__version__,'cabrnet':importlib.metadata.version('cabrnet')},
        'checkpoint_sha256':file_sha256(args.model_dir/'model_state.pth'),
        'strict_load_passed':True,'model_state_unchanged':before==after,
        'score_shape':list(scores.shape),'prototype_map_shape':list(maps.shape),'presence_shape':list(presence.shape),
        'repeat_max_abs_error':float((scores-again).abs().max()),
        'all_classes_reconstruction_max_abs_error':float((scores-reconstructed).abs().max()),
        'inference_seconds_one_forward':forward_seconds,'load_seconds':load_seconds,
        'prototype_threshold':None,'score_semantics':'native raw nonnegative class evidence; not probability; no 0.1 clamp',
        'target':target,'target_name':class_names.get(target),'target_fixed_by_user':args.target is not None,
        'target_score':float(scores[0,target]),'no_evidence_abstention':no_evidence,
        'prediction':None if no_evidence else int(scores[0].argmax()),
        'prediction_name':None if no_evidence else class_names.get(int(scores[0].argmax())),
        'note':'Synthetic checks do not measure bird recognition accuracy. Peak grid cells are not entire receptive fields.',
    }
    (args.output/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
