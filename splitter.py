import os
from pathlib import Path

def split_data(data_path, train_path, test_path, n):
    data_path = Path(data_path)
    train_path = Path(train_path)
    test_path = Path(test_path)
    
    if not train_path.exists():
        train_path.mkdir(parents=True)
    if not test_path.exists():
        test_path.mkdir(parents=True)

    for subject_folder in data_path.iterdir():
        if subject_folder.is_dir():
            train_subject_folder = train_path / subject_folder.name
            test_subject_folder = test_path / subject_folder.name
            if not train_subject_folder.exists():
                train_subject_folder.mkdir()
            if not test_subject_folder.exists():
                test_subject_folder.mkdir()
            
            images = sorted(subject_folder.glob('*.jpg'))
            
            for i, image in enumerate(images[:-n], 1):
                new_name = f"{i}_{image.name}"
                os.rename(str(image), str(train_subject_folder / new_name))
            for i, image in enumerate(images[-n:], 1):
                new_name = f"{i}_{image.name}"
                os.rename(str(image), str(test_subject_folder / new_name))
