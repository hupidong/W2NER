import os
import json
import shutil


def cluener2conll(src_root_path=None, tgt_root_path=None):
    """convert cluener2020 dataset to conll style dataset （bio）"""
    """存在nested形式的实体，无法用conll形式标注，随机舍弃"""
    train_path = os.path.join(src_root_path, 'train.json')
    dev_path = os.path.join(src_root_path, 'dev.json')

    # load original data
    train = load_data(data_path=train_path)
    dev = load_data(data_path=dev_path)

    # transform  to conll-style
    train, _ = json2conll(data=train)
    print('\nin train-dataset，{} entity abandoned。\n'.format(_))
    dev, _ = json2conll(data=dev)
    print('in dev-dataset，{} entity abandoned。\n'.format(_))

    # save
    print('\nsaving dataset:\n')
    save2conll(src=train, save_path=os.path.join(tgt_root_path, 'train.culener.char.bio'))
    save2conll(src=dev, save_path=os.path.join(tgt_root_path, 'dev.cluener.char.bio'))
    print('saving done.\n')
    print('copy dev-dataset to test-dataset.')
    shutil.copyfile(os.path.join(tgt_root_path, 'dev.cluener.char.bio'),
                    os.path.join(tgt_root_path, 'test.cluener.char.bio'))


def load_data(data_path=None):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
        data = [json.loads(sample) for sample in data]
    return data


def json2conll(data=None):
    conll_data_list = []
    abandon_count = 0
    for sample in data:
        sentence = sample['text']
        sentence = list(sentence)
        labels = sample['label']
        tag_list = {}
        for tag_name, tag_value in labels.items():
            b_tag = 'B-' + tag_name.strip().upper()
            i_tag = 'I-' + tag_name.strip().upper()
            for span, index in tag_value.items():
                assert len(span) == (index[0][-1] - index[0][0] + 1)
                flag = False
                for i in range(index[0][0], index[0][-1] + 1):
                    if i in tag_list:
                        flag = True
                        break
                if flag:
                    abandon_count += 1
                    continue
                for i in range(index[0][0], index[0][-1] + 1):
                    assert i not in tag_list
                    if i == index[0][0]:
                        tag_list[i] = b_tag
                    else:
                        tag_list[i] = i_tag
        conll_data = [word + '\t' + tag_list[i] + '\n' if i in tag_list.keys() else word + '\t' + 'O\n' for (i, word) in
                      enumerate(sentence)]
        conll_data_list.append(conll_data)
    return conll_data_list, abandon_count


def save2conll(src=None, save_path=None):
    if not os.path.exists(os.path.split(os.path.realpath(save_path))[0]):
        os.mkdir(os.path.split(os.path.realpath(save_path))[0])
    with open(save_path, 'w', encoding='utf-8') as f:
        for i, sample in enumerate(src):
            f.writelines(sample)
            if i != (len(src) - 1):
                f.writelines('\n')


if __name__ == '__main__':
    cluener2conll(src_root_path='./', tgt_root_path='./')
