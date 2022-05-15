import os
import json
import fastNLP
from fastNLP import Vocabulary
from fastNLP.io.loader import ConllLoader
from fastNLP.core.metrics import _get_encoding_type_from_tag_vocab
from fastNLP.core.metrics import _bmes_tag_to_spans, _bio_tag_to_spans, _bmeso_tag_to_spans, _bioes_tag_to_spans


def convert_data(root_path=None, save_path=None):
    '''
    convert coll-style ner corpus to json-style used in W2NER
    '''
    file_paths = {}
    for file_name in os.listdir(root_path):
        if file_name.startswith('train'):
            file_paths['train'] = os.path.join(root_path, file_name)
        elif file_name.startswith('dev'):
            file_paths['dev'] = os.path.join(root_path, file_name)
        elif file_name.startswith('test'):
            file_paths['test'] = os.path.join(root_path, file_name)

    datasets = {}
    loader = ConllLoader(['char', 'target'])
    for key, file_path in file_paths.items():
        datasets[key] = loader.load(file_path).datasets['train']
    label_vocab = Vocabulary()
    label_vocab.from_dataset(datasets['train'], datasets['dev'], datasets['test'], field_name='target')
    encoding_type = _get_encoding_type_from_tag_vocab(label_vocab)
    print('encoding type is: {}'.format(encoding_type))

    max_seq = 0
    for key, dataset in datasets.items():
        max_len = max([len(seq) for seq in dataset['char']])
        if max_len > max_seq:
            max_seq = max_len
    print('max seq length is: {}'.format(max_seq))

    # convert
    examples = {}
    for key, dataset in datasets.items():
        conll2json(dataset, encoding_type=encoding_type)
        example_list = []
        for sentence, ner in zip(dataset['char'], dataset['span']):
            example_list.append({'sentence': sentence, 'ner': ner})
        examples[key] = example_list

    # write2file
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for key, example in examples.items():
        save_file_path = os.path.join(save_path, key+'.json')
        with open(save_file_path, 'w', encoding='utf-8') as f:
            json.dump(example, f, ensure_ascii=False)


def conll2json(dataset: fastNLP.DataSet, encoding_type=None):
    if encoding_type == 'bmes':  # bmes, bio  bmeso   bioes
        dataset.apply_field(_bmes_tag_to_spans, field_name='target', new_field_name='span')
    elif encoding_type == 'bio':
        dataset.apply_field(_bio_tag_to_spans, field_name='target', new_field_name='span')
    elif encoding_type == 'bmeso':
        dataset.apply_field(_bmeso_tag_to_spans, field_name='target', new_field_name='span')
    elif encoding_type == 'bioes':
        dataset.apply_field(_bioes_tag_to_spans, field_name='target', new_field_name='span')
    dataset.apply_field(convert_fastnlpspan, field_name='span', new_field_name='span')
    return dataset


def convert_fastnlpspan(spans):
    span_list = []
    for span in spans:
        span_dict = {}
        tag = span[0].upper()
        index = list(range(span[1][0], span[1][1]))
        span_dict['index'] = index
        span_dict['type'] = tag
        span_list.append(span_dict)
    return span_list


if __name__ == "__main__":
    data_root_path = r'f:\Work\kg\ner\org'
    data_save_path = r'.\org'
    convert_data(root_path=data_root_path, save_path=data_save_path)
