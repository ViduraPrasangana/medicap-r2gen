import os
from abc import abstractmethod

import time
import torch
import pandas as pd
from numpy import inf
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import wandb


# wandb.init(project="medicap", entity="vidura", resume=True)

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)
        # if args.contrastive is not None:
        #     self._load_contrastive(args.contrastive)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        train_losses = []
        valid_losses = []
        complete_reslts = {}
        best_caption = {}
        print("start train : lr_ve -", self.args.lr_ve)

        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_reslts = {}
            result, result_caption = self._train_epoch(epoch)
            train_losses.append(result["train_loss"])
            valid_losses.append(result["valid_loss"])
            self.plot_diag(train_losses, valid_losses, self.args.save_dir + "/losses.png")
            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                epoch_reslts[str(key)] = value
                print('\t{:15s}: {}'.format(str(key), value))

            complete_reslts[epoch] = epoch_reslts

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                    best_caption = result_caption
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()
        self.__save_json(complete_reslts, 'r2gen_model_train_logs')
        self.__save_json(best_caption, 'r2gen_model_captions')
        print("end r2gen model train")

    def __save_json(self, result, record_name):
        result_path = self.args.record_dir

        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(os.path.join(result_path, '{}.json'.format(record_name)), 'w') as f:
            json.dump(result, f)
        print("logs saved in", result_path)

    def plot_diag(self, train_losses, valid_losses, output):
        plt.plot(train_losses, '-o')
        plt.plot(valid_losses, '-o')
        plt.xlabel('epoch')
        plt.ylabel('losses')
        plt.legend(['Train', 'Valid'])
        plt.title('Train vs Valid Losses')
        plt.savefig(output)
        plt.show()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name + '.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            wandb_best_path = os.path.join(wandb.run.dir, 'model_best.pth')
            torch.save(state, best_path)
            torch.save(state, wandb_best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        try:
            resume_path = str(resume_path)
            print("Loading checkpoint: {} ...".format(resume_path))
            checkpoint = torch.load(resume_path)
            self.start_epoch = checkpoint['epoch'] + 1
            self.mnt_best = checkpoint['monitor_best']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
        except Exception as err:
            print("[Load Checkpoint Failed {}!]\n".format(err))

    def _load_contrastive(self, contrastive_path):
        try:
            contrastive_path = str(contrastive_path)
            print("Loading contrastive model: {} ...".format(contrastive_path))
            checkpoint = torch.load(contrastive_path)
            self.model.load_state_dict(checkpoint['visual_extractor_model'], strict=True)
            print("Contrastive model loaded.")
        except Exception as err:
            print("[Load Checkpoint Failed {}!]\n".format(err))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        wandb.config = {
            "learning_rate": lr_scheduler.get_lr()[0],
            "epochs": args.epochs,
            "batch_size": args.batch_size
        }
        print("train, validation, test -", len(train_dataloader), len(val_dataloader), len(test_dataloader))

    def _train_epoch(self, epoch):

        train_loss = 0
        log = {'train_loss': 0}
        if (self.args.test is None):
            self.model.train()
            iter_wrapper_train = lambda x: tqdm(x, total=len(self.train_dataloader))
            for batch_idx, (images_id, images, reports_ids, reports_masks) in iter_wrapper_train(enumerate(self.train_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)
                output = self.model(images, reports_ids, mode='train')
                # print(output.shape)
                loss = self.criterion(output, reports_ids, reports_masks)
                train_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
            log = {'train_loss': train_loss / len(self.train_dataloader)}

        wandb.watch(self.model)

        valid_loss = 0
        self.model.eval()
        iter_wrapper_valid = lambda x: tqdm(x, total=len(self.val_dataloader))
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in iter_wrapper_valid(enumerate(self.val_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)
                output = self.model(images, reports_ids, mode='train')
                loss = self.criterion(output, reports_ids, reports_masks)
                valid_loss += loss.item()
            log.update(**{'valid_loss': valid_loss / len(self.val_dataloader)})

        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in iter_wrapper_valid(enumerate(self.val_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        result_caption = {}
        self.model.eval()
        iter_wrapper_test = lambda x: tqdm(x, total=len(self.test_dataloader))
        wandb_data = [["epoch_" + str(epoch), "epoch_" + str(epoch), "epoch_" + str(epoch)]]
        with torch.no_grad():
            test_gts, test_res = [], []
            out = [{"epoch": epoch}]
            for batch_idx, (images_id, images, reports_ids, reports_masks) in iter_wrapper_test(enumerate(self.test_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                for i, e in enumerate(reports):
                    dic = {}
                    dic["id"] = images_id[i]
                    dic["predict"] = reports[i]
                    dic["ground_truth"] = ground_truths[i]
                    wandb_data.append([images_id[i], ground_truths[i], reports[i]])
                    out.append(dic)
                test_res.extend(reports)
                test_gts.extend(ground_truths)

                # print(ground_truths)
                # print(reports, "\n")
                # print("each results\n")

                for index in range(len(images_id)):
                    image_id, real_sent, pred_sent = images_id[index], ground_truths[index], reports[index]
                    # print(image_id, "real_sent - ", real_sent, "pred_sent - ",pred_sent)
                    result_caption[image_id] = {
                        'Image id': image_id,
                        'Real Sent': real_sent,
                        'Pred Sent': pred_sent,
                    }
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        columns = ["Id", "Ground truth", "Prediction"]
        wandb_table = wandb.Table(data=wandb_data, columns=columns)
        wandblog = {'results': wandb_table, 'Learning rate': self.lr_scheduler.get_lr()[0],
                    "Train loss": train_loss / len(self.train_dataloader),
                    'Valid loss': valid_loss / len(self.val_dataloader)}
        wandblog.update(**{k: v for k, v in test_met.items()})
        wandb.log(wandblog)
        self.lr_scheduler.step()

        return log, result_caption
