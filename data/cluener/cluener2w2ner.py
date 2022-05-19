import os
import json
import shutil


def cluener2w2ner(src_root_path=None, tgt_root_path=None):
    """convert cluener2020 json-style dataset to w2ner json-style dataset"""
    """w2ner: """
    train_path = os.path.join(src_root_path, 'train.json.raw')
    dev_path = os.path.join(src_root_path, 'dev.json.raw')
    test_path = os.path.join(src_root_path, 'test.json.raw')

    # load original data
    train = load_data(data_path=train_path)
    dev = load_data(data_path=dev_path)
    test = load_data(data_path=test_path)

    # transform
    train = transform2w2ner(src=train)
    dev = transform2w2ner(src=dev)
    test = transform2w2ner(src=test, unlabel=True)

    # save
    print('\nsaving dataset:\n')
    save2w2ner(obj=train, save_path=os.path.join(tgt_root_path, 'train.json'))
    save2w2ner(obj=dev, save_path=os.path.join(tgt_root_path, 'dev.json'))
    save2w2ner(obj=test, save_path=os.path.join(tgt_root_path, 'test.unlabeled.json'))
    print('saving done.\n')
    print('copy dev-dataset to test-dataset.')
    shutil.copyfile(os.path.join(tgt_root_path, 'dev.json'), os.path.join(tgt_root_path, 'test.json'))


def load_data(data_path=None):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
        data = [json.loads(sample) for sample in data]
    return data


def transform2w2ner(src=None, unlabel=False):
    w2ner_list = []
    for sample in src:
        w2ner_sample = {}
        sentence = list(sample['text'])
        w2ner_sample['sentence'] = sentence
        if unlabel:
            w2ner_sample['id'] = sample['id']
        if not unlabel:
            labels = sample['label']
            ner_list = []
            for tag_name, tag_values in labels.items():
                for _, inds in tag_values.items():
                    for ind in inds:
                        ner = {}
                        ner['index'] = list(range(ind[0], ind[-1] + 1))
                        ner['type'] = tag_name.upper()
                        ner_list.append(ner)
            w2ner_sample['ner'] = ner_list
        w2ner_list.append(w2ner_sample)
    return w2ner_list


def save2w2ner(obj, save_path):
    if not os.path.exists(os.path.split(os.path.realpath(save_path))[0]):
        os.mkdir(os.path.split(os.path.realpath(save_path))[0])
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)


if __name__ == '__main__':
    cluener2w2ner(src_root_path='./', tgt_root_path='./')
