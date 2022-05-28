import argparse
import json
import os.path

import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import data_loader
import utils
from model import Model


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]

        self.optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        """
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=config.warm_factor * updates_total,
                                                                      num_training_steps=updates_total)

        self.optimizer = torch.optim.SGD(params, lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9)
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=[config.bert_learning_rate * 2,
                                                                                     config.bert_learning_rate * 2,
                                                                                     config.learning_rate * 2],
                                                             total_steps=updates_total)
        

        self.scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=self.optimizer,
                                                           base_lr=[config.bert_learning_rate * 0.1,
                                                                    config.bert_learning_rate * 0.1,
                                                                    config.learning_rate * 0.1],
                                                           max_lr=[config.bert_learning_rate,
                                                                   config.bert_learning_rate,
                                                                   config.learning_rate],
                                                           step_size_up=len(train_loader),
                                                           step_size_down=len(train_loader),
                                                           cycle_momentum=False)
        """
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer,
                                                                              T_0=2, T_mult=2,
                                                                              eta_min=0)


    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []
        iters = len(data_loader)
        logger.info(f"one train-epoch has {iters} iterations.")
        for i, data_batch in enumerate(tqdm(data_loader)):
            data_batch = [data.cuda() for data in data_batch[:-1]]

            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

            outputs = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)

            grid_mask2d = grid_mask2d.clone()
            loss = self.criterion(outputs[grid_mask2d], grid_labels[grid_mask2d])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_list.append(loss.cpu().item())

            outputs = torch.argmax(outputs, -1)
            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs = outputs[grid_mask2d].contiguous().view(-1)

            label_result.append(grid_labels.cpu())
            pred_result.append(outputs.cpu())
            if config.debug:
                logger.info(
                    f"epoch: {epoch}, batch: {i}, lr: {self.optimizer.param_groups[0]['lr']} \t {self.optimizer.param_groups[1]['lr']} \t {self.optimizer.param_groups[2]['lr']}")

            self.scheduler.step(epoch + i / iters)
            #self.scheduler.step()

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")

        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
                      ["{:3.4f}".format(x) for x in [f1, p, r]])
        logger.info("\n{}".format(table))
        return f1

    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()

        pred_result = []
        label_result = []

        total_ent_real = 0
        total_ent_pred = 0
        total_ent_cross = 0
        with torch.no_grad():
            for i, data_batch in enumerate(tqdm(data_loader)):
                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                ent_cross, ent_pred, ent_real, _ = utils.decode(outputs.cpu().numpy(), entity_text,
                                                                length.cpu().numpy())

                total_ent_real += ent_real
                total_ent_pred += ent_pred
                total_ent_cross += ent_cross

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.cpu().numpy(),
                                                      pred_result.cpu().numpy(),
                                                      average="macro")  # 此处是word-word-pair-relation的评价,不是label的评价
        ent_f1, ent_p, ent_r = utils.cal_f1(total_ent_cross, total_ent_pred, total_ent_real)

        title = "EVAL" if not is_test else "TEST"
        logger.info('{} Label F1 {}'.format(title, f1_score(label_result.cpu().numpy(),
                                                            pred_result.cpu().numpy(),
                                                            average=None)))

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [ent_f1, ent_p, ent_r]])

        logger.info("\n{}".format(table))
        return ent_f1

    def predict(self, data_loader, data):
        self.model.eval()

        result = []

        i = 0
        with torch.no_grad():
            for data_batch in tqdm(data_loader):
                sentence_batch = data[i:i + config.batch_size]
                entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, decode_entities = utils.decode(outputs.cpu().numpy(), entity_text,
                                                                    length.cpu().numpy())

                for ent_list, sentence in zip(decode_entities, sentence_batch):
                    sentence = sentence["sentence"]
                    instance = {"sentence": sentence, "entity": []}
                    for ent in ent_list:
                        instance["entity"].append({"text": [sentence[x] for x in ent[0]],
                                                   "index": ent[0],
                                                   "type": config.vocab.id_to_label(ent[1])})
                    result.append(instance)

                i += config.batch_size

        with open(config.predict_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

        return result

    """
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
    """

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--predict_path', type=str)
    parser.add_argument('--device', type=int)

    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)

    parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)

    parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")

    parser.add_argument('--seed', type=int)
    parser.add_argument('--pred_unlabeled_data', action='store_true')
    parser.add_argument('--choose_by', type=str, default='dev', choices=['dev', 'test'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--n_jobs', type=int, default=1)

    args = parser.parse_args()

    config = config.Config(args)

    save_root_path = os.path.realpath(os.path.join('./models', config.dataset))
    config.save_root_path = save_root_path
    if not os.path.exists(save_root_path):
        os.mkdir(save_root_path)
    logger = utils.get_logger(config.dataset)
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    # random.seed(config.seed)
    # np.random.seed(config.seed)
    # torch.manual_seed(config.seed)
    # torch.cuda.manual_seed(config.seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    logger.info("Loading Data")
    datasets, ori_data = data_loader.load_data_bert(config)

    train_loader, dev_loader, test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=config.batch_size,
                   collate_fn=data_loader.collate_fn,
                   shuffle=i == 0,
                   num_workers=0,
                   drop_last=(i == 0))
        for i, dataset in enumerate(datasets)
    )

    updates_total = len(train_loader) * config.epochs

    logger.info("Building Model")
    model = Model(config)

    model = model.cuda()
    total = [param.nelement() for param in model.parameters()]
    logger.info('Number of parameters: {}'.format(sum(total)))

    trainer = Trainer(model)

    best_dev_f1 = 0
    best_test_f1 = 0
    for i in range(config.epochs):
        logger.info("Epoch: {}".format(i))
        trainer.train(i, train_loader)
        dev_f1 = trainer.eval(i, dev_loader)
        test_f1 = trainer.eval(i, test_loader, is_test=True)
        if args.choose_by == 'dev':
            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_test_f1 = test_f1
                trainer.save(os.path.join(config.save_root_path, 'model.pt'))
        elif args.choose_by == 'test':
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                best_dev_f1 = dev_f1
                trainer.save(os.path.join(config.save_root_path, 'model.pt'))
    logger.info('choose best model by {} dataset, default by dev dataset.'.format(args.choose_by))
    logger.info("Best DEV F1: {:3.4f}".format(best_dev_f1))
    logger.info("Best TEST F1: {:3.4f}".format(best_test_f1))
    trainer.load(os.path.join(config.save_root_path, 'model.pt'))
    trainer.eval("Final", test_loader, is_test=True)

    if args.pred_unlabeled_data:
        from data_loader import load_data_bert_predict, collate_fn_predict
        import config_predict
        from predict import Predictor

        logger.info('Predicting unlabeled dataset.')
        config_pred = config_predict.Config('config/cluener_predict.json')
        config_pred.model = os.path.join('models/{}'.format(config.dataset), 'model.pt')
        config_pred.tokenizer = os.path.join('models/{}'.format(config.dataset), 'tokenizer.pt')
        config_pred.vocab = os.path.join('models/{}'.format(config.dataset), 'vocab.pt')
        with open('data/cluener/test.json.raw', 'r', encoding='utf-8') as f:
            instances = f.readlines()
            instances = [json.loads(text) for text in instances]
            texts = [instance['text'] for instance in instances]
            ids = [instance['id'] for instance in instances]

        pred_dataset = load_data_bert_predict(texts=texts, config=config_pred)
        pred_loader = DataLoader(dataset=pred_dataset,
                                 batch_size=config.batch_size,
                                 collate_fn=collate_fn_predict,
                                 shuffle=False,
                                 num_workers=2,
                                 drop_last=False)
        logger.info('model form {} will be loaded.'.format(config_pred.model))
        predictor = Predictor(config_pred, args.device)
        preds = predictor.predict(pred_loader)
        preds = utils.w2ner2cluener(src=preds, ids=ids, debug=True)
        save_path = os.path.join('models/{}'.format(config.dataset), 'output.' + config.dataset + '.unlabeled.json')
        logger.info('Save path is {}'.format(save_path))
        with open(save_path, 'w', encoding='utf-8') as f:
            for i, pred in enumerate(preds):
                pred = json.dumps(pred, ensure_ascii=False)
                if i != (len(preds) - 1):
                    f.write(pred), f.write('\n')
                else:
                    f.write(pred)
        logger.info('Predicted data saved.')
