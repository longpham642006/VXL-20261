"""Download and verify the PhysioNet 1.0.0 releases used by the project."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
DATABASES = {
    "mitdb": "mit-bih-arrhythmia-database",
    "nstdb": "mit-bih-noise-stress-test-database",
}


def verify(directory):
    manifest = directory / "SHA256SUMS.txt"
    count = 0
    for line in manifest.read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        path = directory / filename.lstrip("*")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"SHA-256 sai: {path}")
        count += 1
    if not count:
        raise ValueError(f"Danh sách checksum rỗng: {manifest}")
    return count


def install(name, archive, staging):
    folder = DATABASES[name] + "-1.0.0"
    with ZipFile(archive) as release:
        release.extractall(staging)
    source = staging / folder
    verify(source)
    target = RAW / name
    for path in source.rglob("*"):
        if path.is_file():
            dest = target / path.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            path.replace(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Kiểm tra ngoại tuyến")
    parser.add_argument("--archive-dir", type=Path, help="Thư mục ZIP có tên mitdb.zip và nstdb.zip")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for name, slug in DATABASES.items():
        if not args.verify:
            with tempfile.TemporaryDirectory(dir=RAW) as tmp:
                staging = Path(tmp)
                if args.archive_dir:
                    archive = args.archive_dir / f"{name}.zip"
                else:
                    archive = staging / "release.zip"
                    url = f"https://physionet.org/static/published-projects/{name}/{slug}-1.0.0.zip"
                    with urlopen(url, timeout=120) as response, archive.open("wb") as out:
                        shutil.copyfileobj(response, out)
                install(name, archive, staging)
        count = verify(RAW / name)
        print(f"{name}: SHA-256 OK ({count} tệp)")


if __name__ == "__main__":
    main()
