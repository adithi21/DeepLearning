from ..model import getModel
from ..data import Cifar10Data
from tqdm import trange
import torch
from pathlib import Path
from functools import partial
from torch.optim import lr_scheduler

def check_for_dir(*args_path):
    for path in args_path:
        if not path.exists():
            path.mkdir(parents=True)

def delete_file(path):
    if path.exists() and not path.is_dir():
        path.unlink()

def logger(filepath , *args,**kwargs):
    print(*args,**kwargs)
    with open(filepath,"a") as f:  # appends to file and closes it when finished
        print(file=f,*args,**kwargs)

def save_checkpoint(epoch, epochs_since_improvement, model, optimizer,loss, best_loss, is_best,filepath=None):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement
    :param model: model
    :param optimizer: optimizer
    :param loss: validation loss in this epoch
    :param best_loss: best validation loss achieved so far (not necessarily in this checkpoint)
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'best_loss': best_loss,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    filename = Path("./src/saved_weights/") if filepath is None else filepath
    is_best = is_best if filepath is None else False
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, Path("./src/saved_weights/"))

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: CrossEntropy loss
    :param optimizer :optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    losses = AverageMeter()  # loss
    accuracy = AverageMeter() # Accuracy meter

    # Batches
    for images, labels in train_loader:
      
        optimizer.zero_grad()
        # Move to default device
        images = images.to(device) 
        labels = labels.to(device)

        # Forward prop.
        output = model(images)  
        preds = torch.argmax(output,1)
        # Loss
        loss = criterion(output, labels)  # scalar

        
        loss.backward()
        
        
        optimizer.step()

        losses.update(loss.item(), images.shape[0])
        accuracy.update(torch.sum(preds == labels.data).item())
        
        # Print status
    print(f"TRAIN  Loss {losses.avg}\t Accuracy {accuracy.sum/len(train_loader.dataset)}")
    return losses.avg

def train_model(model,
          data_loader ,
          optimizer=None,
          criterion =None,
          num_epochs=5 ,
          save_model_filename="saved_weights.pt",
          log_filename="training_logs.txt"):
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,weight_decay=weight_decay,momentum=0.9,nesterov=True,dampening=0)  #model.parameters(), lr=1e-3)
    if criterion is None:
        criterion =torch.nn.CrossEntropyLoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=4,min_lr=1e-5,verbose=True)
    global logger
    log_filename_path = Path("./src/training_logs/")
    save_model_filename_path = Path("./src/saved_weights/")
    check_for_dir(log_filename_path,save_model_filename_path)
    save_model_filename_path = save_model_filename_path/save_model_filename
    log_filename_path = log_filename_path/log_filename
    delete_file(log_filename_path)
    logger = partial(logger,log_filename_path)    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_val_loss = float("inf")
    for epoch in trange(num_epochs,desc="Epochs"):
        print("Current LR is " , optimizer.param_groups[0]['lr'])
      # One epoch's training
      train_loss = train(train_loader=trainLoader,
                         model=model,
                         criterion=criterion,
                         optimizer=optimizer,
                         epoch=epoch)

      # One epoch's validation
      val_loss = validate(val_loader=valLoader,
                          model=model,
                          criterion=criterion)
      scheduler.step(val_loss)
      # Did validation loss improve?
      is_best = val_loss < best_loss
      best_loss = min(val_loss, best_loss)

      if not is_best:
          epochs_since_improvement += 1
          print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))

      else:
          epochs_since_improvement = 0

      # Save checkpoint
      save_checkpoint(epoch, epochs_since_improvement, model, optimizer, train_loss, best_loss, is_best)
    return model


