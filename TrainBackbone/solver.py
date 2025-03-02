from dataset import PetDataset
from torch.utils.data import DataLoader
import torch
from net import Net
import os
import copy
import csv
from utils import calculate_iou
import time

class Solver():
    def __init__(self, args):
        self.args = args

        # prepare a dataset
        mean = [0.47964323, 0.4472308,  0.39698297]
        std = [0.23054191, 0.22776432, 0.22883151]
        self.train_data = PetDataset(train=True,
                                  data_root=args.data_root,
                                  size=args.image_size,
                                  mean = mean,
                                  std = std)
        self.test_data = PetDataset(train=False,
                                  data_root=args.data_root,
                                  size=args.image_size,
                                  mean = mean,
                                  std = std)
        self.train_loader = DataLoader(dataset=self.train_data,
                                       batch_size=args.batch_size,
                                       num_workers=4,
                                       shuffle=True, drop_last=True)
        
        print(len(self.train_data))
        
        # turn on the CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # build the net
        self.net = Net(2).to(self.device)

        # define loss function and optimizer
        self.classLossFunc = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=0.0001)

        # create checkpoint directory
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)

    def fit(self):
        args = self.args

        self.loss_values = list()
        self.train_acc_values = list()
        self.test_acc_values = list()

        classifier_stopped = False

        early_stopping_patience = 4

        log = ""
        start_time = time.time()


        # loop over epochs
        for epoch in range(args.max_epochs):
            self.net.train() # set the net in training mode

            # loop over the training set
            for (images, labels) in self.train_loader:
                # load batch in gpu
                images = images.to(self.device)
                labels = labels.to(self.device)

                # perform a forward pass
                predictions = self.net(images)
                classLoss = self.classLossFunc(predictions, labels)
                totalLoss = classLoss

                # perform a backward pass
                self.optim.zero_grad()
                totalLoss.backward()
                self.optim.step()
                
                # print loss value every step
                log_item = "Epoch [{}/{}] Loss: {:.3f} ".format(epoch + 1, args.max_epochs, totalLoss.item())
                print(log_item)
                log += log_item + "\n"

            # evaluate model and save checkpoints
            if (epoch+1) % args.print_every == 0:
                # evaluate and print accuracy and iou
                train_acc = self.evaluate(self.train_data)
                test_acc  = self.evaluate(self.test_data)

                log_item = "Epoch [{}/{}] Loss: {:.3f} Train Acc: {:.3f}, Test Acc: {:.3f}, Time: {}".\
                    format(epoch+1, args.max_epochs, totalLoss.item(), train_acc, test_acc, time.time() - start_time)
                print(log_item)
                log += log_item + "\n"
                
                # save loss, accuracy and IoU for plotting
                self.loss_values.append(float(totalLoss))
                self.train_acc_values.append(float(train_acc))
                self.test_acc_values.append(float(test_acc))

                # save checkpoint
                self.save(args.ckpt_dir, args.ckpt_name, epoch+1)

            # check for early stopping condition
            #   train accuracy increasing for 2 epochs AND test accuracy decreasing for 2 epochs => stop classifier training
            #   train IoU increasing for 2 epochs AND test IoU decreasing for 2 epochs => stop regressor training
            stop_classifier = self.early_stopping(epoch, early_stopping_patience)
            
            if stop_classifier and not classifier_stopped:
                log_item = f"Classifier training stopped at epoch {epoch} due to early stopping condition"
                print(log_item)
                log += log_item + "\n"
                classifier_stopped = True
            
            # early stopping condition reached for both classifier and regressor => stop training
            if classifier_stopped:
                break  

        # save loss, accuracy and IoU to a csv file
        data = zip(self.loss_values, self.train_acc_values, self.test_acc_values)
        with open('training_results.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in data:
                writer.writerow(row)

        # save log file
        with open("log.txt", "w") as file:
            file.write(log)

    def evaluate(self, data):
        args = self.args
        loader = DataLoader(data,
                            batch_size=args.batch_size,
                            num_workers=1,
                            shuffle=False)

        self.net.eval() # set the model in evaluation mode
        num_correct, num_total = 0, 0
        iou = 0

        # switch off autograd
        with torch.no_grad():
            # loop over the validation set
            for (images, labels) in loader:
                # send the input to the device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # make the predictions
                predictions = self.net(images)

                # calculate the number of correct predictions
                num_correct += (predictions.argmax(1) == labels).type(
                    torch.float).sum().item()
                num_total += labels.size(0)

        return num_correct / num_total
    
    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)

    def early_stopping(self, e, patience):
        stop_classifier = False

        if(e>patience-1):
            train_acc_improving = True
            test_acc_worsening = True
            
            for i in range(0, patience): 
                train_acc_improving = train_acc_improving and (self.train_acc_values[e-i] > self.train_acc_values[e-i-1])
                test_acc_worsening = test_acc_worsening and (self.test_acc_values[e-1-i] > self.test_acc_values[e-i])
            
            stop_classifier = train_acc_improving and test_acc_worsening

        return stop_classifier