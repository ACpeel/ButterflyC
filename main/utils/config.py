import os
import shutil
import yaml

REQUIRED_DATASET_ENTRIES = (
    'Training_set.csv',
    'Testing_set.csv',
    'train',
    'test',
)

def __docs__():
    '''
    device: "CPU"
    num_classes: 75
    image_size: 224
    batch_size: 16
    capacity: 250
    precision: "float16"
    learning_rate: 0.001
    epochs: 100
    static: "app/templates/static"
    source_path: "data"
    model_path: "main/models"
    log_dir: "main/model/log"
    data: "TEMP/data"
    upload_dir: "TEMP/upload"
    train_csv: "TEMP/data/Training_set.csv"
    test_csv: "TEMP/data/Testing_set.csv"
    train_data: "TEMP/data/train"
    test_data: "TEMP/data/test"
    '''


def dataset_layout_ready(dataset_root):
    return all(
        os.path.exists(os.path.join(dataset_root, entry))
        for entry in REQUIRED_DATASET_ENTRIES
    )


def sync_dataset_tree(source_root, target_root):
    os.makedirs(target_root, exist_ok=True)

    for entry in REQUIRED_DATASET_ENTRIES:
        source = os.path.join(source_root, entry)
        target = os.path.join(target_root, entry)

        if not os.path.exists(source):
            continue

        if os.path.isdir(source):
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)

def load_config(config_file='config.yml')->dict:
    with open(config_file,'r') as f:
        configs = yaml.safe_load(f)
    # ensure dirs
    source_ready = dataset_layout_ready(configs['source_path'])
    target_ready = dataset_layout_ready(configs['data'])

    if not target_ready:
        if source_ready:
            print(
                f'同步数据集目录：{configs["source_path"]} -> {configs["data"]}'
            )
            sync_dataset_tree(configs['source_path'], configs['data'])
        else:
            print(f'数据集不存在or未准备好：{configs["source_path"]}')
            os.makedirs(configs['data'], exist_ok=True)

    managed_dirs = [
        configs['model_path'],
        configs['log_dir'],
        configs['upload_dir'],
        os.path.dirname(configs.get('manifest_path', '')),
        os.path.dirname(configs.get('labels_path', '')),
    ]

    for directory in managed_dirs:
        if directory:
            os.makedirs(directory, exist_ok=True)

    return configs

if __name__=='__main__':
    load_config()
