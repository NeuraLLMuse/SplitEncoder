import torch
import torch.nn.functional as F
from torcheval.metrics.text import Perplexity
import random
from tqdm import tqdm
from metrics_utils import compute_normalized_token_entropy
import random
import csv
import numpy as np
import os
from transformers import get_cosine_schedule_with_warmup
import sys

perplexity_metric = Perplexity(ignore_index=-100)

def full_to_partial_masking(
        harmony_tokens,
        mask_token_id,
        num_visible=0,
        bar_token_id=None
    ):
    """
    Generate visible input and denoising target for diffusion-style training.

    Args:
        harmony_tokens (torch.Tensor): Tensor of shape (B, L) containing target harmony token ids.
        stage (int): Current training stage (0 to total_stages - 1).
        total_stages (int): Total number of diffusion stages.
        mask_token_id (int): The token ID used to mask hidden positions in visible_harmony.
        device (str or torch.device): Target device.

    Returns:
        visible_harmony (torch.Tensor): Tensor of shape (B, L) with visible tokens (others masked).
        denoising_target (torch.Tensor): Tensor of shape (B, L) with tokens to predict (others = -100).
    """
    device = harmony_tokens.device
    B, L = harmony_tokens.shape

    visible_harmony = torch.full_like(harmony_tokens, fill_value=mask_token_id)
    denoising_target = torch.full_like(harmony_tokens, fill_value=-100)  # -100 is ignored by CrossEntropyLoss

    if bar_token_id is not None:
        # Create a mask for bar token positions
        bar_mask = (harmony_tokens == bar_token_id)
        # Put bar tokens in visible_harmony (always unmasked)
        visible_harmony[bar_mask] = bar_token_id
        # # Also include them in the denoising target (so model predicts them too)
        denoising_target[bar_mask] = bar_token_id
    
    perm = torch.randperm(L, device=device)

    visible_idx = perm[:num_visible]
    predict_idx = perm[num_visible:]  # predict all remaining
    # print('visible_idx: ', visible_idx)
    # print('predict_idx: ', predict_idx)

    visible_harmony[:, visible_idx] = harmony_tokens[:, visible_idx]
    denoising_target[:, predict_idx] = harmony_tokens[:, predict_idx]

    return visible_harmony, denoising_target
# end full_to_partial_masking

def validation_f2f2f(model, valloader, mask_token_id, bar_token_id, \
                    num_visible, loss_fn, epoch, step, \
                    train_loss, train_accuracy, \
                    train_perplexity, train_token_entropy,
                    best_val_loss, saving_version, results_path=None, \
                    transformer_path=None, tqdm_position=0, save_every_epoch=True,
                    epochs_to_save=None):
    device = model.device
    model.eval()
    with torch.no_grad():
        val_loss = 0
        running_loss = 0
        batch_num = 0
        running_accuracy = 0
        val_accuracy = 0
        running_perplexity = 0
        val_perplexity = 0
        running_token_entropy = 0
        val_token_entropy = 0
        print('validation')
        with tqdm(valloader, unit='batch', position=tqdm_position) as tepoch:
            tepoch.set_description(f'Epoch {epoch}@{step}| val')
            for batch in tepoch:
                perplexity_metric.reset()
                melody_grid = batch["pianoroll"].to(device)           # (B, 256, 140)
                harmony_gt = batch["harmony_ids"].to(device)         # (B, 256)
                
                # Apply masking to harmony
                harmony_input, harmony_target = full_to_partial_masking(
                    harmony_gt,
                    mask_token_id,
                    num_visible,
                    bar_token_id=bar_token_id
                )
                
                # Forward pass
                logits = model(
                    melody_grid.to(device),
                    harmony_input.to(device),
                )

                # Compute loss only on masked tokens
                loss = loss_fn(logits.view(-1, logits.size(-1)), harmony_target.view(-1))

                # update loss and accuracy
                batch_num += 1
                running_loss += loss.item()
                val_loss = running_loss/batch_num
                # accuracy
                predictions = logits.argmax(dim=-1)
                mask = harmony_target != -100
                running_accuracy += (predictions[mask] == harmony_target[mask]).sum().item()/mask.sum().item()
                val_accuracy = running_accuracy/batch_num
                # perplexity
                running_perplexity += perplexity_metric.update(logits, harmony_target).compute().item()
                val_perplexity = running_perplexity/batch_num
                # token entropy
                _, entropy_per_batch = compute_normalized_token_entropy(logits, harmony_target, pad_token_id=-100)
                running_token_entropy += entropy_per_batch
                val_token_entropy = running_token_entropy/batch_num

                tepoch.set_postfix(loss=val_loss, accuracy=val_accuracy)
            # end for batch
    # end with tqdm
    if transformer_path is not None:
        if  save_every_epoch or (best_val_loss > val_loss):
            print('saving!')
            saving_version += 1
            best_val_loss = val_loss
            torch.save(model.state_dict(), transformer_path)
        if epochs_to_save is not None:
            if  epoch in epochs_to_save:
                print(f'saving copy of epoch {epoch} with num_visible {num_visible}')
                torch.save(model.state_dict(), transformer_path.replace('.pt', f'_epoch{epoch}_nvis{num_visible}.pt'))
    print(f'validation: accuracy={val_accuracy}, loss={val_loss}')
    print('results_path: ', results_path)
    if results_path is not None:
        with open( results_path, 'a' ) as f:
            writer = csv.writer(f)
            writer.writerow( [epoch, step, num_visible, train_loss, train_accuracy, \
                            train_perplexity, train_token_entropy, \
                            val_loss, val_accuracy, \
                            val_perplexity, val_token_entropy, \
                            saving_version] )
    return best_val_loss, saving_version
# end validation_loop

def train_f2f2f(
    model, optimizer, trainloader, valloader, loss_fn, mask_token_id,
    epochs=100,
    exponent=5,
    results_path=None,
    transformer_path=None,
    bar_token_id=None,
    validations_per_epoch=1,
    tqdm_position=0,
    save_every_epoch=True,
    epochs_to_save=None
):
    # device = next(model.parameters()).device
    device = model.device
    perplexity_metric.to(device)
    best_val_loss = np.inf
    saving_version = 0

    # save results and model
    print('results_path:', results_path)
    if results_path is not None:
        result_fields = ['epoch', 'step', 'n_vis', 'train_loss', 'train_acc', \
                        'train_ppl', 'train_te', 'val_loss', \
                        'val_acc', 'val_ppl', 'val_te', 'sav_version']
        with open( results_path, 'w' ) as f:
            writer = csv.writer(f)
            writer.writerow( result_fields )

    # Compute total training steps
    total_steps = len(trainloader) * epochs
    # Define the scheduler
    # warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    step = 0

    for epoch in range(epochs):
        train_loss = 0
        running_loss = 0
        batch_num = 0
        running_accuracy = 0
        train_accuracy = 0
        running_perplexity = 0
        train_perplexity = 0
        running_token_entropy = 0
        train_token_entropy = 0
        
        with tqdm(trainloader, unit='batch', position=tqdm_position) as tepoch:
            tepoch.set_description(f'Epoch {epoch}@{step} | trn')
            for batch in tepoch:
                perplexity_metric.reset()
                model.train()
                melody_grid = batch["pianoroll"].to(device)    # (B, L, prDim)
                harmony_gt = batch["harmony_ids"].to(device)     # (B, L)

                # Apply masking to harmony
                if exponent == -1:
                    percent_visible = 0.0
                else:
                    # percent_visible = min(1.0, (step+1)/total_steps)**exponent  # 5th power goes around half way near zero
                    percent_visible = min(1.0, np.exp( (-1/2)*np.power( (step-(total_steps/2))/(total_steps/15) , 2) ))
                L = harmony_gt.shape[1]
                num_visible = min( int(L * percent_visible), L-1 )  # ensure at least one token is predicted
                harmony_input, harmony_target = full_to_partial_masking(
                    harmony_gt,
                    mask_token_id,
                    num_visible,
                    bar_token_id=bar_token_id
                )

                # Forward pass
                logits = model(
                    melody_grid.to(device),
                    harmony_input.to(device),
                )

                # Compute loss only on masked tokens
                loss = loss_fn(logits.view(-1, logits.size(-1)), harmony_target.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()

                # update loss and accuracy
                batch_num += 1
                running_loss += loss.item()
                train_loss = running_loss/batch_num
                # accuracy
                predictions = logits.argmax(dim=-1)
                mask = harmony_target != -100
                running_accuracy += (predictions[mask] == harmony_target[mask]).sum().item()/max(1,mask.sum().item())
                train_accuracy = running_accuracy/batch_num
                # perplexity
                running_perplexity += perplexity_metric.update(logits, harmony_target).compute().item()
                train_perplexity = running_perplexity/batch_num
                # token entropy
                _, entropy_per_batch = compute_normalized_token_entropy(logits, harmony_target, pad_token_id=-100)
                running_token_entropy += entropy_per_batch
                train_token_entropy = running_token_entropy/batch_num

                tepoch.set_postfix(loss=train_loss, accuracy=train_accuracy)
                step += 1
                if step%(total_steps//(epochs*validations_per_epoch)) == 0 or step == total_steps:
                    best_val_loss, saving_version = validation_f2f2f(
                        model,
                        valloader,
                        mask_token_id,
                        bar_token_id,
                        num_visible,
                        loss_fn,
                        epoch,
                        step,
                        train_loss,
                        train_accuracy,
                        train_perplexity,
                        train_token_entropy,
                        best_val_loss,
                        saving_version,
                        results_path=results_path,
                        transformer_path=transformer_path,
                        tqdm_position=tqdm_position,
                        save_every_epoch=save_every_epoch,
                        epochs_to_save=epochs_to_save
                    )
            # end for batch
        # end with tqdm
    # end for epoch
# end train_with_curriculum
