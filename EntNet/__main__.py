#!/usr/bin/env python3
import os

from tensorboardX import SummaryWriter
import torch
from torch.utils import data as data_utils

from dataset import bAbIDataset
from model.EntNet import REN
from model.entnet_as_in_parlai import RecurrentEntityNetwork
from model.LSTM import LSTM
from Utils.Argparse import parse
from Utils import Visdom
from Utils.Saver import Saver, GlobalVarSaver, StateSaver


# TODO: Check if every tensor is moved to the right cuda device.
def main():
    global epoch
    # Get arguments, setup,  prepare data and print some info
    args = parse()

    log_path = os.path.join("logs", args.name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    if args.task == 'babi':
        train_dataset = bAbIDataset(args.dataset_path, args.babi_task)
        val_dataset = bAbIDataset(args.dataset_path, args.babi_task, train=False)
    else:
        raise NotImplementedError

    # Setting up the Model
    if args.model == 'lstm':
        model = LSTM(40, train_dataset.num_vocab, 100, args.device,
                     sentence_size=max(train_dataset.sentence_size, train_dataset.query_size))
        print("Using LSTM")
    else:
        # model = REN(args.num_blocks, train_dataset.num_vocab, 100, args.device, train_dataset.sentence_size,
        #             train_dataset.query_size).to(args.device)
        model = RecurrentEntityNetwork(train_dataset.num_vocab, device=args.device,
                                       sequence_length=max(train_dataset.sentence_size, train_dataset.query_size))
        print("Using EntNet")
    if args.multi:  # TODO: Whats this?
        model = torch.nn.DataParallel(model, device_ids=args.gpu_range)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        Exception("Invalid optimizer")
    if args.cyc_lr:
        cycle_momentum = True if args.optimizer == 'sgd' else False
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 5e-5, args.lr, cycle_momentum=cycle_momentum, step_size_up=args.cyc_step_size_up)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    # Before we are getting started, let's get ready to give some feedback
    print("Dataset size: ", len(train_dataset))
    print("Sentence size:", train_dataset.sentence_size)
    print("Vocab set", [str(i)+': '+str(train_dataset.vocab[i]) for i in range(len(train_dataset.vocab))])

    # Prepare Visdom
    Visdom.start()
    lr_plt = Visdom.Plot2D("Curent learning rate", store_interval=1, xlabel="Epochs", ylabel="Learning Rate")
    # TODO: Check legend
    train_loss = Visdom.Plot2D("Loss on Train Data", store_interval=1, xlabel="iteration", ylabel="loss", legend=['one', 2, 'three'])
    train_accuracy = Visdom.Plot2D("Accuracy on Train Data", store_interval=1, xlabel="iteration", ylabel="accuracy")
    validation_loss = Visdom.Plot2D("Loss on Validation Set", store_interval=1, xlabel="epoch", ylabel="loss")
    validation_accuracy = Visdom.Plot2D("Accuracy on Validation Set", store_interval=1, xlabel="epoch", ylabel="accuracy")
    babi_text_plt = Visdom.Text("Network Output")
    train_plots = {'loss': train_loss, 'accuracy': train_accuracy}
    val_plots = {'text': babi_text_plt}

    epoch = 0

    # Register Variables and plots to save
    saver = Saver(os.path.join(args.output_path, args.name), short_interval=args.save_interval)
    saver.register('train_loss', StateSaver(train_loss))
    saver.register('train_accuracy', StateSaver(train_accuracy))
    saver.register('validation_loss', StateSaver(validation_loss))
    saver.register('validation_accuracy', StateSaver(validation_accuracy))
    saver.register('lr_plot', StateSaver(lr_plt))
    saver.register("model", StateSaver(model))
    saver.register("optimizer", StateSaver(optimizer))
    saver.register("epoch", GlobalVarSaver('epoch'))
    # saver.register("train_dataset", StateSaver(train_dataset))
    # saver.register("val_dataset", StateSaver(val_dataset))

    eval_on_start = False
    print("Given model argument to load from: ", args.load_model)
    # TODO: Load learning rate scheduler
    if args.load_model:
        if not saver.load(args.load_model):
            #  model.reset_parameters()
            print('Not loading, something went wrong', args.load_model)
            pass
        else:
            eval_on_start = False

    start_epoch = epoch
    end_epoch = start_epoch + args.epochs
    model.to(args.device)

    # TODO: Use saver only on full epochs or use it on certain iteration

    """ TRAIN START """
    # Eval on Start
    if eval_on_start:
        val_result = val_dataset.eval(args, model, plots=val_plots)
        validation_loss.add_point(0, val_result['loss'])
        validation_accuracy.add_point(0, val_result['accuracy'])
        saver.write(epoch)
    for epoch in range(start_epoch, end_epoch):
        train_result = train_dataset.test(args, model, optimizer, epoch=epoch, plots=train_plots, scheduler=lr_scheduler)
        val_result = val_dataset.eval(args, model, epoch=epoch+1, plots=val_plots)
        validation_loss.add_point(epoch, val_result['loss'])
        validation_accuracy.add_point(epoch, val_result['accuracy'])

        current_lr = None
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            break
        lr_plt.add_point(epoch, current_lr if current_lr else 0)

        saver.tick(epoch+1)
        if not args.cyc_lr:
            lr_scheduler.step()

        # TODO: Add writer
        # Log
        if epoch % args.save_interval == 0 or epoch == args.epochs-1:
            for param_group in optimizer.param_groups:
                log_lr = param_group['lr']
                break

            log = 'Epoch: [{epoch}]\t Train Loss {tl} Acc {ta}\t Val Loss {vl} Acc {va} lr {lr}'.format(
                epoch=epoch, tl=round(train_result['loss'], 3), ta=round(train_result['accuracy'], 3),
                vl=round(val_result['loss'], 3), va=round(val_result['accuracy'], 3), lr=log_lr)
            print(log)


if __name__ == '__main__':
    main()
