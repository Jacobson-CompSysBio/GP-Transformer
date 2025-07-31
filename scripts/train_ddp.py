# imports
import time
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModel, 
    AutoModelForSequenceClassification, 
    AutoModelForMaskedLM, 
    AutoTokenizer
    )
from transformers.models.bert.configuration_bert import BertConfig

# non-customizable options
iter_update = 'train loss {1:.4e}, val loss {2:.4e}\r'
best_val_loss = None # initialize best validation loss
last_improved = 0 # start early stopping counter
iter_num = 0 # initialize iteration counter
t0 = time.time() # start timer

# training loop
# refresh log
with open(log_path, 'w') as f: 
    f.write(f'iter_num,train_loss,val_loss\n')

# keep training until break
while True:

    # clear print output
    clear_output(wait=True)

    if best_val_loss is not None:
        print('---------------------------------------\n',
            f'Iteration: {iter_num} | Best Loss: {best_val_loss:.4e}\n', 
            '---------------------------------------', sep = '')
    else:
        print('-------------\n',
            f'Iteration: {iter_num}\n', 
            '-------------', sep = '')

    #
    # checkpoint
    #

    # shuffle dataloaders
    train_loader = DataLoader(
        train_generator, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True)
    
    val_loader = DataLoader(
        val_generator, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True)

    # estimate loss
    model.eval()
    with torch.no_grad():
        train_loss, val_loss = 0, 0
        with tqdm(total=batches_per_eval, desc='Eval') as pbar:
            for (xbt, ybt), (xbv, ybv) in zip(train_loader, val_loader):
                xbt, ybt = xbt.to(device), ybt.to(device)
                xbv, ybv = xbv.to(device), ybv.to(device)
                train_loss += loss_function(model(xbt), ybt).item()
                val_loss += loss_function(model(xbv), ybv).item()
                pbar.update(1)
                if pbar.n == pbar.total:
                    break
        train_loss /= batches_per_eval
        val_loss /= batches_per_eval
    model.train()

    # update user
    print(iter_update.format(iter_num, train_loss, val_loss)) 

    # update log
    with open(log_path, 'a') as f: 
        f.write(f'{iter_num},{train_loss},{val_loss}\n')

    # checkpoint model
    if iter_num > 0:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'kwargs': model_kwargs,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'train_ids': train_idx,
            'val_ids': val_idx,
        }
        torch.save(checkpoint, chckpnt_path.format(iter_num))

    # book keeping
    if best_val_loss is None:
        best_val_loss = val_loss

    if iter_num > 0:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            last_improved = 0
            print(f'*** validation loss improved: {best_val_loss:.4e} ***')
        else:
            last_improved += 1
            print(f'validation has not improved in {last_improved} steps')
        if last_improved > early_stop:
            print()
            print(f'*** no improvement for {early_stop} steps, stopping ***')
            break

    # --------
    # backprop
    # --------

    # shuffle dataloaders
    train_loader = DataLoader(
        train_generator, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True)

    # iterate over batches
    with tqdm(total=eval_interval, desc='Train') as pbar:
        for xb, yb in train_loader:

            # update the model
            xb, yb = xb.to(device), yb.to(device)

            loss = loss_function(model(xb), yb)

            if torch.isnan(loss):
                print('loss is NaN, stopping')
                break
            
            # apply learning rate schedule
            lr = get_lr(it = iter_num,
                        warmup_iters = warmup_iters, 
                        lr_decay_iters = lr_decay_iters, 
                        max_lr = max_lr, 
                        min_lr = min_lr)
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # update book keeping
            pbar.update(1)
            iter_num += 1
            if pbar.n == pbar.total:
                break

    # break once hitting max_iters
    if iter_num > max_iters:
        print(f'maximum iterations reached: {max_iters}')
        break

