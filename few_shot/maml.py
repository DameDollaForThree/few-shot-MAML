from select import select
import torch
from collections import OrderedDict
from torch.optim import Optimizer
from torch.nn import Module
from typing import Dict, List, Callable, Union

from few_shot.core import create_nshot_task_label

def replace_grad(parameter_gradients, parameter_name):
    def replace_grad_(module):
        return parameter_gradients[parameter_name]

    return replace_grad_

# I added another argument 'other_optim'
def meta_gradient_step(model: Module,
                       optimiser: Optimizer,
                       loss_fn: Callable,
                       x: torch.Tensor,
                       y: torch.Tensor,
                       n_shot: int,
                       k_way: int,
                       q_queries: int,
                       order: int,
                       inner_train_steps: int,
                       inner_lr: float,
                       train: bool,
                       device: Union[str, torch.device],
                       other_optim: List[Optimizer] = None,  # added by me
                       p_task: List[int] = None,  # added by me
                       p_meta: List[int] = None):  # added by me
    """
    Perform a gradient step on a meta-learner.

    # Arguments
        model: Base model of the meta-learner being trained
        optimiser: Optimiser to calculate gradient step from loss  
        loss_fn: Loss function to calculate between predictions and outputs
        x: Input samples for all few shot tasks
        y: Input labels of all few shot tasks
        n_shot: Number of examples per class in the support set of each task
        k_way: Number of classes in the few shot classification task of each task
        q_queries: Number of examples per class in the query set of each task. The query set is used to calculate
            meta-gradients after applying the update to
        order: Whether to use 1st order MAML (update meta-learner weights with gradients of the updated weights on the
            query set) or 2nd order MAML (use 2nd order updates by differentiating through the gradients of the updated
            weights on the query with respect to the original weights).
        inner_train_steps: Number of gradient steps to fit the fast weights during each inner update
        inner_lr: Learning rate used to update the fast weights on the inner update
        train: Whether to update the meta-learner weights at the end of the episode.
        device: Device on which to run computation
    """
    if other_optim != None:
        meta_conv_optim, meta_other_optim = other_optim
    
    data_shape = x.shape[2:]
    create_graph = (True if order == 2 else False) and train

    task_gradients = []
    task_losses = []
    task_predictions = []
    # iterate through each task in the meta batch
    for j, meta_batch in enumerate(x):
        # By construction x is a 5D tensor of shape: (meta_batch_size, n*k + q*k, channels, width, height)
        # Hence when we iterate over the first  dimension we are iterating through the meta batches
        x_task_train = meta_batch[:n_shot * k_way]
        x_task_val = meta_batch[n_shot * k_way:]

        # Create a fast model using the current meta model weights
        fast_weights = OrderedDict(model.named_parameters())
        
        # Train the model for `inner_train_steps` iterations
        for inner_batch in range(inner_train_steps):
            # Perform update of model weights
            y = create_nshot_task_label(k_way, n_shot).to(device)  # TODO: pseudo-label???
            logits = model.functional_forward(x_task_train, fast_weights)
            loss = loss_fn(logits, y)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
            
            ###### filter selection based on batchnorm ###### 
            if train:
                # Update weights manually: first, only update batch_norm and FC weights!!!
                fast_weights = OrderedDict(
                    (name, param - inner_lr * grad) if name[0:5] == 'other' else (name, param)
                    for ((name, param), grad) in zip(fast_weights.items(), gradients)
                )
                
                with torch.no_grad():
                    # compute mask according to bn_gammas and p_list
                    masks = OrderedDict()
                    for i in range(4):
                        bn_gammas = fast_weights[f'other_param.{i*2}']
                        num_filter = bn_gammas.shape[0]
                        selected = torch.topk(bn_gammas, int(num_filter * p_task[i-1]))[1]
                        masks[f'conv_param.{i*2}'] = torch.zeros_like(fast_weights[f'conv_param.{i*2}'])
                        masks[f'conv_param.{i*2}'][selected, :, :, :] = 1
                        masks[f'conv_param.{i*2+1}'] = torch.zeros_like(fast_weights[f'conv_param.{i*2+1}'])
                        masks[f'conv_param.{i*2+1}'][selected] = 1
                    
                    # zero mask for other params
                    for i in range(4):
                        bn_gammas = fast_weights[f'other_param.{i*2}']
                        masks[f'other_param.{i*2}'] = torch.zeros(bn_gammas.shape[0])
                        masks[f'other_param.{i*2+1}'] = torch.zeros(bn_gammas.shape[0])
                    masks['other_param.8'] = torch.zeros_like(fast_weights['other_param.8'])
                    masks['other_param.9'] = torch.zeros_like(fast_weights['other_param.9'])
                
                # Update weights manually: second, updates conv layers with masks!!!
                fast_weights = OrderedDict(
                    (name, param - inner_lr * grad * mask.cuda()) if name[0:5] != 'other' else (name, param)
                    for ((name, param), grad, (_, mask)) in zip(fast_weights.items(), gradients, masks.items())
                )
                
            else:
                fast_weights = OrderedDict(
                    (name, param - inner_lr * grad)
                    for ((name, param), grad) in zip(fast_weights.items(), gradients)
                )
            
        # Do a pass of the model on the validation data from the current task
        y = create_nshot_task_label(k_way, q_queries).to(device)
        logits = model.functional_forward(x_task_val, fast_weights)
        loss = loss_fn(logits, y)
        # loss.backward(retain_graph=True)  # backward(): computes grad, doesn't update/.step() --> not necessary...

        # Get post-update accuracies
        y_pred = logits.softmax(dim=1)
        task_predictions.append(y_pred)

        # Accumulate losses and gradients
        task_losses.append(loss)
        gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
        named_grads = {name: g for ((name, _), g) in zip(fast_weights.items(), gradients)}
        task_gradients.append(named_grads)

    # I didn't use order == 1.
    if order == 1:
        if train:
            sum_task_gradients = {k: torch.stack([grad[k] for grad in task_gradients]).mean(dim=0)
                                  for k in task_gradients[0].keys()}
            hooks = []
            for name, param in model.named_parameters():
                hooks.append(
                    param.register_hook(replace_grad(sum_task_gradients, name))
                )

            model.train()
            optimiser.zero_grad()
            # Dummy pass in order to create `loss` variable
            # Replace dummy gradients with mean task gradients using hooks
            logits = model(torch.zeros((k_way, ) + data_shape).to(device, dtype=torch.double))
            loss = loss_fn(logits, create_nshot_task_label(k_way, 1).to(device))
            loss.backward()
            optimiser.step()

            for h in hooks:
                h.remove()

        return torch.stack(task_losses).mean(), torch.cat(task_predictions)

    # I use order == 2.
    elif order == 2:
        model.train()  # set the model in train mode
        optimiser.zero_grad()
        meta_batch_loss = torch.stack(task_losses).mean()   
            
        if train:
            meta_conv_optim.zero_grad()
            meta_other_optim.zero_grad()
            meta_batch_loss.backward()
            meta_other_optim.step()
            
            with torch.no_grad():
                    # compute and apply mask according to other_param and p_list
                    for i in range(4):
                        bn_gammas = model.other_param[i*2]
                        conv_weight = model.conv_param[i*2]
                        conv_bias = model.conv_param[i*2 + 1]
                        
                        num_filter = bn_gammas.shape[0]
                        selected = torch.topk(bn_gammas, int(num_filter * p_meta[i-1]))[1]
                        
                        conv_weight_mask = torch.zeros_like(conv_weight)
                        conv_weight_mask[selected, :, :, :] = 1
                        conv_weight.grad *= conv_weight_mask
                        
                        conv_bias_mask = torch.zeros_like(conv_bias)
                        conv_bias_mask[selected] = 1
                        conv_bias.grad *= conv_bias_mask
    
            meta_conv_optim.step()

        return meta_batch_loss, torch.cat(task_predictions)
    else:
        raise ValueError('Order must be either 1 or 2.')
