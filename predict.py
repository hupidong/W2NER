import argparse
import torch
import torch.autograd
from torch.utils.data import DataLoader

import data_loader
import utils
from model import Model


class Predictor(object):
    def __init__(self, config, device):
        self.model = torch.load(config.model)
        self.vocab = torch.load(config.vocab)
        self.model.to(device)
        self.device = device

    def predict(self, data_loader):
        self.model.eval()
        predict_result = []
        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                texts = data_batch[-1]
                data_batch = [data.to(self.device) for data in data_batch[:-1]]
                bert_inputs, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch
                outputs = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                outputs = torch.argmax(outputs, -1)
                instances = predict_decode(outputs.cpu().numpy(), sent_length.cpu().numpy(), texts, self.vocab)
                for instance in instances:
                    result = {'sentence': instance[0], "entity":[]}
                    if len(instance) == 1:
                        predict_result.append(result)
                    else:
                        for entity in instance[1:]:
                            result['entity'].append({"text": entity[0],
                                                   "index": [entity[2],entity[3]],
                                                   "type": entity[1]})
                        predict_result.append(result)
        return predict_result


def predict_decode(outputs, length, texts, vocab):
    entities = []
    for index, (instance, l, text) in enumerate(zip(outputs, length, texts)):
        forward_dict = {}
        head_dict = {}
        ht_type_dict = {}
        for i in range(l):
            for j in range(i + 1, l):
                if instance[i, j] == 1:
                    if i not in forward_dict:
                        forward_dict[i] = [j]
                    else:
                        forward_dict[i].append(j)
        for i in range(l):
            for j in range(i, l):
                if instance[j, i] > 1:
                    ht_type_dict[(i, j)] = instance[j, i]
                    if i not in head_dict:
                        head_dict[i] = {j}
                    else:
                        head_dict[i].add(j)

        predicts = []

        def find_entity(key, entity, tails):
            entity.append(key)
            if key not in forward_dict:
                if key in tails:
                    predicts.append(entity.copy())
                entity.pop()
                return
            else:
                if key in tails:
                    predicts.append(entity.copy())
            for k in forward_dict[key]:
                find_entity(k, entity, tails)
            entity.pop()

        def convert_index_to_text(index, type):
            text = "-".join([str(i) for i in index])
            text = text + "-#-{}".format(type)
            return text

        for head in head_dict:
            find_entity(head, [], head_dict[head])
        predicts = set([convert_index_to_text(x, ht_type_dict[(x[0], x[-1])]) for x in predicts])
        tmp = (text,)
        for pre in predicts:
            pre = pre.split('-#-')
            print(pre)
            print(text)
            ind = pre[0].split('-')
            entity = text[int(ind[0]):int(ind[-1]) + 1]
            entity_type = vocab.id2label[int(pre[1])]
            tmp += ((entity, entity_type, int(ind[0]), int(ind[-1])),)
        entities.append(tmp)
    return entities


if __name__ == '__main__':
    import config_predict
    config = config_predict.Config('config/cluener_predict.json')
    logger = utils.get_logger('predict')
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device('cuda:0')

    logger.info("Loading Data")
    texts = [
        "高勇，男，中国国籍，无境外居留权。",
        "常见量，男。"
    ]
    # 这一步要在model之前创建，因为还有给config添加属性
    predict_dataset = data_loader.load_data_bert_predict(texts, config)
    predict_loader = DataLoader(dataset=predict_dataset,
                                batch_size=config.batch_size,
                                collate_fn=data_loader.collate_fn_predict,
                                shuffle=False,
                                num_workers=2,
                                drop_last=False)
    # updates_total这个参数直接设置为0即可
    updates_total = 0
    logger.info("Building Model")
    model = Model(config)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    predictor = Predictor(model, config, device)
    predictor.predict(predict_loader)
