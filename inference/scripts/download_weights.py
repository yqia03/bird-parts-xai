"""Download one published CUB/ConvNeXt PIP-Net model, not the 2.5 GB bundle.

Uses public HTTP byte ranges and ZIP CRC validation. Python 3.10+, standard
library only. This is a CaBRNet-trained checkpoint, NOT Nauta's original weights.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen
import zipfile

RECORD = 'https://zenodo.org/records/18429182'
URL = RECORD + '/files/pipnet_cub200_stanfordcars.zip?download=1'
SIZE = 2477145362
PREFIX = 'pipnet_cub200_convnext_s0/final/'
REPO_ROOT = Path(__file__).resolve().parents[1]
NAMES = ['model_arch.yml', 'dataset.yml', 'training.yml',
         'projection_info.csv', 'model_state.pth']


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.pos = 0
        self.bytes_received = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else SIZE + offset
        if not 0 <= self.pos <= SIZE:
            raise ValueError('seek out of archive bounds')
        return self.pos

    def read(self, n=-1):
        n = SIZE - self.pos if n is None or n < 0 else min(n, SIZE - self.pos)
        if not n:
            return b''
        if n > 150_000_000:
            raise RuntimeError('Unexpected large range; refusing full archive download')
        start, end = self.pos, self.pos + n - 1
        request = Request(URL, headers={'Range': f'bytes={start}-{end}',
                                       'User-Agent': 'CS5495-checkpoint-verification/1.0'})
        for attempt in range(3):
            try:
                with urlopen(request, timeout=120) as response:
                    expected = f'bytes {start}-{end}/{SIZE}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise RuntimeError('Server did not return the requested byte range; '
                                           'use the full ZIP download from the record page instead')
                    data = response.read(n + 1)
                if len(data) != n:
                    raise RuntimeError('Incomplete byte range')
                self.pos += n
                self.bytes_received += n
                return data
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO_ROOT / 'weights' / 'cabrnet_cub_s0_final')
    parser.add_argument('--configs-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = [n for n in NAMES if not args.configs_only or n != 'model_state.pth']
    records = []
    remote = RemoteZip()
    with zipfile.ZipFile(remote) as archive:
        for name in names + ['../log.txt']:
            member = PREFIX + name if name != '../log.txt' else 'pipnet_cub200_convnext_s0/log.txt'
            destination = args.output / Path(name).name
            info = archive.getinfo(member)
            print(f'Downloading {member} ({info.compress_size:,} compressed bytes)', flush=True)
            data = archive.read(info)  # ZipFile verifies CRC before returning.
            destination.write_bytes(data)
            records.append({'member':member, 'local_file':destination.name,
                            'bytes':len(data), 'archive_crc32':f'{info.CRC:08x}',
                            'sha256':hashlib.sha256(data).hexdigest()})
    manifest = {'source': RECORD, 'archive_url': URL,
                'published_archive_md5': '131986538f9a63d736c05f64988fa3aa',
                'verification': 'selected members CRC32 checked; full archive MD5 not checked',
                'model_identity': 'CaBRNet-trained PIPNet / CUB200 / ConvNeXt / seed0 / final',
                'not_original_pipnet_weights': True,
                'downloaded_bytes':remote.bytes_received, 'files':records}
    (args.output/'download_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Done. Received {remote.bytes_received:,} bytes. Files: {args.output.resolve()}')


if __name__ == '__main__':
    main()
