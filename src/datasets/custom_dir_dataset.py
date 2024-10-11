from pathlib import Path

from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    def __init__(self, dir, *args, **kwargs):
        data = []
        for path in Path(dir / 'audio').iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)
                if Path(dir / 'transcriptions' / (path.stem + ".txt")).exists():
                    transc_path = Path(dir / 'transcriptions' / (path.stem + ".txt"))
                    if transc_path.exists():
                        with transc_path.open() as f:
                            entry["text"] = f.read().strip()
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
