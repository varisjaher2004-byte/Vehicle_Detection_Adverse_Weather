import pandas as pd
from pathlib import Path


def save_csv(data, save_path):

    save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(data)

    df.to_csv(save_path, index=False)

    print(f"CSV Saved : {save_path}")