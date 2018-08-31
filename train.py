
"""
Train and test the chosen model.
File should be run from the commandline so arguments can be passed.
"""

import os
import random
import shutil
from datetime import datetime
import timeit

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchnet as tnt
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from tqdm import tqdm

import train_cfg
from capsule_utils import Decoder
from pytorch_train_utils import EarlyStopping


def reset_meters():
    """
    Reset the loss and accuracy meters.
    """
    all_loss_meter.reset()
    accuracy_meter.reset()
    if args.remake:
        class_loss_meter.reset()
        recon_loss_meter.reset()


def save_checkpoint(state, is_best, folder):
    """
    Save the current state of the model, optimizer,
    scheduler, and decoder to a loadable checkpoint file.
    """
    if not os.path.exists(os.path.dirname(folder)):
        os.makedirs(os.path.dirname(folder))
    ckpnt_path = os.path.join(folder, 'checkpoint.pth.tar')
    torch.save(state, ckpnt_path)
    if is_best:
        copy_path = os.path.join(folder, 'model_best.pth.tar')
        shutil.copy(ckpnt_path, copy_path)


def log_reconstruction(remake, original, niter, epoch):
    """
    Save the reconstructed image to the chosen subdirectory.
    """
    remake = remake.view(-1, *args.input_shape)
    path = os.path.join(subdir_path, args.save_data_dir)
    if not os.path.exists(path):
        os.makedirs(path)

    vutils.save_image(original, '%s/original_samples.png_epoch_%03d.png'
                      % (path, epoch), normalize=True)
    writer.add_image('original_samples', vutils.make_grid(original, normalize=True), niter)
    vutils.save_image(remake.detach(), '%s/reconstructed_samples_epoch_%03d.png'
                      % (path, epoch), normalize=True)
    writer.add_image('reconstructions', vutils.make_grid(remake.detach(), normalize=True), niter)


def train(train_loader, model, optimizer, epoch, decoder=None, decoder_optimizer=None):
    """
    Train the model.
    """
    reset_meters()
    time_meter = tnt.meter.AverageValueMeter()
    model.train()

    steps = len(train_loader)
    with tqdm(total=steps) as pbar:
        for i, (data, target) in enumerate(train_loader):
            start = timeit.default_timer()

            data, target = data.to(args.device), target.to(args.device)
            target = target.squeeze(-1)

            output, poses = model(data)
            remake = decoder(poses, target, args.num_classes) if args.remake else None

            all_loss, class_loss, recon_loss = criterion(output, target, remake, data)

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            if decoder_optimizer:
                decoder_optimizer.zero_grad()

            all_loss.backward()

            optimizer.step()
            if decoder_optimizer:
                decoder_optimizer.step()

            accuracy_meter.add(output.detach(), target.detach())
            all_loss_meter.add(all_loss.item())
            time_meter.add(1 / (timeit.default_timer() - start))

            # if i % 25 == 0:
            #     print("{:.2f}".format(time_meter.value()[0]))
            pbar.set_postfix(loss=all_loss_meter.value()[0], accuracy=accuracy_meter.value(1))
            pbar.update()

            # Save scalar data to tensorboard
            if i % args.log_interval == 0:
                niter = (epoch * steps) + i
                writer.add_scalar('Train/All_loss', all_loss_meter.val, niter)
                if args.remake:
                    writer.add_scalar('Train/Class_loss', class_loss, niter)
                    writer.add_scalar('Train/Recon_loss', recon_loss, niter)
                writer.add_scalar('Train/Prec@1', accuracy_meter.value(1), niter)
                writer.add_scalar('Train/Prec@5', accuracy_meter.value(5), niter)
    print("Average iteration time: {:.2f}".format(time_meter.value()[0]))


def test(val_loader, train_loader, model, epoch, decoder=None):
    """
    Test the model.
    """
    reset_meters()
    # switch to evaluate mode
    model.eval()

    niter = epoch * len(train_loader)
    with torch.no_grad():
        steps = len(val_loader)
        with tqdm(total=steps) as pbar:
            for i, (data, target) in enumerate(val_loader):

                data, target = data.to(args.device), target.to(args.device)
                target = target.squeeze(-1)

                # compute output
                output, poses = model(data)
                remake = decoder(poses, target, args.num_classes) if args.remake else None
                all_loss, class_loss, recon_loss = criterion(output, target, remake, data)

                accuracy_meter.add(output.detach(), target.detach())
                all_loss_meter.add(all_loss.item())
                if args.remake:
                    class_loss_meter.add(class_loss.item())
                    recon_loss_meter.add(recon_loss.item())

                pbar.set_postfix(loss=all_loss_meter.value()[0], accuracy=accuracy_meter.value()[0])
                pbar.update()

                # if args.remake and i % args.remake_log_interval == 0:
                #     log_reconstruction(remake, data, niter, epoch - 1)

    top1, top5 = accuracy_meter.value()
    if not args.evaluate:
        writer.add_scalar('Test/All_loss', all_loss_meter.value()[0], niter)
        writer.add_scalar('Test/Prec@1', top1, niter)
        writer.add_scalar('Test/Prec@5', top5, niter)

        if args.remake:
            writer.add_scalar('Test/Class_loss', class_loss_meter.value()[0], niter)
            writer.add_scalar('Test/Recon_loss', recon_loss_meter.value()[0], niter)

    print(' * Prec@1 {top1:.3f} Prec@5 {top5:.3f}'
          .format(top1=top1, top5=top5))

    return top1, all_loss_meter.value()[0]


def main():
    global args, writer, best_prec1, \
        criterion, all_loss_meter, accuracy_meter,\
        subdir_path

    parser = train_cfg.arguments()
    args = parser.parse_args()

    if args.resume and not args.subdir:
        subdir_path = os.path.dirname(args.resume)
    else:
        if not args.subdir:
            args.subdir = " ".join((args.model,  args.dataset, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
        subdir_path = os.path.join(args.root, args.subdir)

    writer = SummaryWriter(os.path.join(subdir_path, args.summary_dir))

    # device setup
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 10000)

    writer.add_text('settings', str(args), 0)
    with open(os.path.join(subdir_path, 'settings'), "w") as file:
        file.write(str(args))

    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.manual_seed)

    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')

    print("=> Using device:", args.device)

    # datasets
    print('=> Loading dataset:', args.dataset)
    train_loader, test_loader = train_cfg.get_dataloaders(args)

    if args.remake_log_interval is None:
        args.remake_log_interval = (len(test_loader) - 2)

    # create model

    print("=> Creating model:", args.model)

    if args.model == 'dense':
        model = train_cfg.model_dict[args.model.lower()](num_classes=args.num_classes).to(args.device)
    else:
        model = train_cfg.model_dict[args.model.lower()](input_shape=args.input_shape,
                                                         num_classes=args.num_classes,
                                                         num_routing=args.routing).to(args.device)

    if args.remake:

        output_atoms = 16
        if args.model == 'spectral':
            output_atoms = model.output_caps.p_size
        if args.model == 'em-one':
            output_atoms = model.class_caps.p_size
        elif args.model == 'dynamic':
            output_atoms = model.digit_caps.output_atoms

        decoder = Decoder(args.num_classes * output_atoms,
                          original_shape=args.input_shape).to(args.device)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001, weight_decay=0)
        decoder_scheduler = optim.lr_scheduler.ExponentialLR(decoder_optimizer, gamma=0.93)
    else:
        decoder = None
        decoder_optimizer = None
        decoder_scheduler = None

    criterion = train_cfg.loss_functions[args.loss](
        args.num_classes, args.remake).to(args.device)

    optim_kwargs = {'nesterov': not args.no_nesterov, 'momentum': args.momentum}\
        if args.optimizer == 'sgd' else {}
    model_optimizer = train_cfg.optimizers[args.optimizer](model.parameters(),
                                                           lr=args.lr,
                                                           weight_decay=args.weight_decay,
                                                           **optim_kwargs)

    model_scheduler = optim.lr_scheduler.ExponentialLR(model_optimizer, gamma=0.93)
    early_stopping = EarlyStopping(patience=args.early_stopping_rounds)

    all_loss_meter = tnt.meter.AverageValueMeter()
    if args.remake:
        global class_loss_meter, recon_loss_meter
        class_loss_meter = tnt.meter.AverageValueMeter()
        recon_loss_meter = tnt.meter.AverageValueMeter()

    accuracy_meter = tnt.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            model_optimizer.load_state_dict(checkpoint['model_optimizer'])
            model_scheduler.last_epoch = args.start_epoch
            early_stopping.best = checkpoint['best']
            early_stopping.num_bad_epochs = checkpoint['num_bad_epochs']
            if args.remake:
                decoder.load_state_dict(checkpoint['decoder'])
                decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
                decoder_scheduler.last_epoch = args.start_epoch
            print("=> Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> No checkpoint found at '{}'".format(args.resume))

    print("=> Total number of parameters:",
          sum(param.numel() for param in model.parameters()))

    # Log the model parameters
    with open(os.path.join(subdir_path, "parameters.txt"), "w") as file:
        file.write(str(model.named_parameters))

    if args.evaluate:
        print("=> Evaluating model")
        test(test_loader, train_loader, model, args.start_epoch, decoder)
        return

    if not args.resume:
        print("=> Testing untrained model")
        best_prec1, _ = test(test_loader, train_loader, model, args.start_epoch, decoder)
        # best_prec1 = 0
    for epoch in range(args.start_epoch, args.epochs):

        print("=> Training on epoch %d" % epoch)
        train(train_loader, model, model_optimizer, epoch, decoder, decoder_optimizer)

        print("=> Testing on epoch %d" % epoch)
        test_prec1, test_loss = test(test_loader, train_loader, model, epoch + 1, decoder)

        # remember best prec@1 and save checkpoint
        is_best = test_prec1 > best_prec1
        if is_best:
            print("=> Test accuracy increased to", test_prec1)
        best_prec1 = max(test_prec1, best_prec1)
        with open(os.path.join(subdir_path, "last_prec1.txt"), "w") as file:
            file.write(str(test_prec1))

        if epoch % args.save_interval == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model,
                'decoder': decoder.state_dict() if args.remake else {},
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'model_optimizer': model_optimizer.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict() if args.remake else {},
                'best': early_stopping.best,
                'num_bad_epochs': early_stopping.num_bad_epochs},
                is_best,
                subdir_path)

        # for name, param in model.named_parameters():
        #         writer.add_histogram(name, param, epoch)

        model_scheduler.step(test_loss)
        if args.remake:
            decoder_scheduler.step(test_loss)

        # early_stop = early_stopping.step(test_loss)
        # if early_stopping.num_bad_epochs > 0:
        #     print('=> Number of epochs without improvement: {}/{}'.
        #           format(early_stopping.num_bad_epochs, early_stopping.patience))
        # if early_stop:
        #     print("=> No validation improvement for %d consecutive epochs. "
        #           "Early stopping triggered." % early_stopping.patience)
        #     break

        # if args.model == 'spectral':
        #     model.anneal_eta(model.eta_0 * 0.005)

        if args.loss == 'spread_loss':
            criterion.increment_margin()


if __name__ == '__main__':
    main()
