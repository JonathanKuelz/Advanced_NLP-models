"""
Dataset class adapted from https://github.com/nmhkahn/MemN2N-pytorch/blob/master/memn2n/dataset.py
"""
import time
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils import data as torch_data_utils
from data_utils import *  # load_task, vectorize_data


class bAbIDataset(data.Dataset):
    def __init__(self, dataset_dir, task_id=1, memory_size=70, train=True):
        self.train = train
        self.task_id = task_id
        self.dataset_dir = dataset_dir

        train_data, test_data = load_task(self.dataset_dir, task_id)
        data = train_data + test_data


        if task_id == 'qa3':
            truncated_story_length = 130
        else:
            truncated_story_length = 70
        stories_train = truncate_stories(train_data, truncated_story_length)
        stories_test = truncate_stories(test_data, truncated_story_length)

        self.vocab, token_to_id = get_tokenizer(stories_train + stories_test)
        self.num_vocab = len(self.vocab)
        self.get_word_from_id = {i: self.vocab[i] for i in range(len(self.vocab))}

        stories_token_train = tokenize_stories(stories_train, token_to_id)
        stories_token_test = tokenize_stories(stories_test, token_to_id)
        stories_token_all = stories_token_train + stories_token_test

        story_lengths = [len(sentence) for story, _, _ in stories_token_all for sentence in story]
        max_sentence_length = max(story_lengths)
        max_story_length = max([len(story) for story, _, _ in stories_token_all])
        max_query_length = max([len(query) for _, query, _ in stories_token_all])
        self.sentence_size = max_sentence_length
        self.query_size = max_query_length
        if train:
            story, query, answer = pad_stories(stories_token_train, \
                max_sentence_length, max_story_length, max_query_length)
        else:

            story, query, answer = pad_stories(stories_token_test, \
                max_sentence_length, max_story_length, max_query_length)
        self.data_story = torch.LongTensor(story)
        self.data_query = torch.LongTensor(query)
        self.data_answer = torch.LongTensor(answer)

    def __getitem__(self, idx):
        return self.data_story[idx], self.data_query[idx], self.data_answer[idx]

    def __len__(self):
        return len(self.data_story)

    def loss(self, net_out, target):
        raise NotImplementedError
        return 0

    def test(self, args, model, optimizer, *, epoch=1, plots=None, report_interval=100, scheduler=None):

        model.train()  # Setting the model to train mode
        loss_crit = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

        loader = torch_data_utils.DataLoader(self, batch_size=args.batchsize, num_workers=args.njobs,
                                             shuffle=True, pin_memory=True, timeout=300, drop_last=True)

        i = 0
        epoch_loss = 0
        epoch_correct = 0
        iter_time = time.time()
        report_loss = 0
        report_accuracy = 0
        previous_iterations = epoch * len(loader)
        for i, (story, query, answer) in enumerate(loader):  # Iterate over all samples
            # Story Shape =  [Batchsize, max_story_lenght, max_sentence_length]

            story, query, answer = story.to(args.device), query.to(args.device), answer.to(args.device)
            model.zero_grad()
            for param in model.parameters():
                param.detach()

            model_out = model(story, query)

            loss = loss_crit(model_out, answer)
            loss.backward(retain_graph=True)  # Retain Graph

            # TODO: No gradient clipping so far
            optimizer.step()
            if scheduler:
                if args.cyc_lr:
                    scheduler.step()

            correct = torch.argmax(model_out.detach(), dim=1).eq(answer.detach()).sum().to("cpu").item()
            epoch_loss = epoch_loss + loss.item()
            epoch_correct += correct
            report_loss = report_loss + loss.item()
            report_accuracy += correct

            if i % report_interval == 0:
                if i == 0:
                    continue
                print('Epoch: {e}, Iteration: {i}, s/iter: {t}'.format(e=epoch+1, i=i, t=(time.time() - iter_time) / report_interval, l=loss))
                iter_time = time.time()

                if plots:
                    if 'loss' in plots:
                        plots['loss'].add_point(i + previous_iterations, report_loss / report_interval)
                    if 'accuracy' in plots:
                        plots['accuracy'].add_point(i + previous_iterations, report_accuracy / (story.shape[0] * report_interval))
                report_loss = 0
                report_accuracy = 0

        return {'loss': epoch_loss / (i + 1), 'accuracy': epoch_correct / ((i + 1) * args.batchsize)}

    def eval(self, args, model, *, epoch=1, plots=None):

        model.eval()  # Setting the model to evaluation mode
        loss_crit = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

        loader = torch_data_utils.DataLoader(self, batch_size=args.batchsize, num_workers=args.njobs,
                                             shuffle=True, pin_memory=True, timeout=300, drop_last=True)

        epoch_loss = 0
        epoch_correct = 0

        with torch.no_grad():
            i = 0
            for i, (story, query, answer) in enumerate(loader):  # Iterate over all samples
                # Story Shape =  [Batchsize, max_story_lenght, max_sentence_length]

                model.zero_grad()
                story, query, answer = story.to(args.device), query.to(args.device), answer.to(args.device)

                model_out = model(story, query)

                epoch_loss += loss_crit(model_out, answer).item()  # loss = -log(softmax(model_out[answer]))
                epoch_correct += torch.argmax(model_out.detach(), dim=1).eq(answer.detach()).sum().to("cpu").item()

        if plots:
            if 'text' in plots:
                preview = plots['text']
                in_txt = ''
                for i, batch in enumerate(story.tolist()):
                    in_txt += '<b>Input</b>'

                    # Input
                    for j, line in enumerate(batch):
                        line_empty = True
                        line_txt = ''
                        for word in line:
                            if word != 0:
                                line_empty = False
                                line_txt += ' ' + self.get_word_from_id[word]
                        if not line_empty:
                            in_txt += line_txt + '<br>'

                        # Query
                        if j == 9:
                            query_txt = '<b>Query</b>'
                            this_query = query[i]
                            for word_id in this_query.tolist():
                                word = self.get_word_from_id[word_id] if word_id != 0 else ''
                                query_txt += ' ' + word
                            in_txt += query_txt + '<br>'

                    # Output and ground truth
                    prediction = F.softmax(model_out[i], dim=0)
                    top_vals, top_idx = torch.topk(prediction, 3)
                    for j, idx in enumerate(top_idx.tolist()):
                        out_pred = self.get_word_from_id[idx] if idx < self.num_vocab else 'internal cell'
                        in_txt += '\n<b>Ground Truth: </b>' + self.get_word_from_id[answer[i].tolist()] + \
                                  ', ' + '<b>Answer {}:</b> {}, '.format(j + 1, out_pred)
                        in_txt += '<b>value</b>: {}<br>'.format(top_vals.tolist()[j])
                    if i == 3:
                        break
                    in_txt += '<br><br>'

                preview.set(in_txt)

        return {'loss': epoch_loss / (len(loader) + 1), 'accuracy': epoch_correct / ((len(loader) + 1) * args.batchsize)}

