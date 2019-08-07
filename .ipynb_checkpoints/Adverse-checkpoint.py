# Misc
import random
import numpy as np
import pandas as pd
import keras

# Pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd.gradcheck import zero_gradients

def clip(current, low_bound, up_bound):
    assert(len(current) == len(up_bound) and len(low_bound) == len(up_bound))
    low_bound = torch.FloatTensor(low_bound)
    up_bound = torch.FloatTensor(up_bound)
    clipped = torch.max(torch.min(current, up_bound), low_bound)
    return clipped




def lowProFool(x_old, model, weights, bounds, maxiters, alpha, lambda_, overshoot=0.002):
    r = Variable(torch.FloatTensor(1e-4 * np.ones(x_old.numpy().shape)), requires_grad=True) 
    v = torch.FloatTensor(np.array(weights))
    
    output = model.forward(x_old + r)
    orig_pred = output.max(0, keepdim=True)[1].cpu().numpy() # get the index of the max log-probability
    target_pred = np.abs(1 - orig_pred)
    
    if orig_pred == 0:
        origin = Variable(torch.tensor([1., 0.], requires_grad=False))
        target = Variable(torch.tensor([0., 1.], requires_grad=False))
    else:
        origin = Variable(torch.tensor([0., 1.], requires_grad=False))
        target = Variable(torch.tensor([1., 0.], requires_grad=False))
    
    lambda_ = torch.tensor([lambda_])
    
    bce = torch.nn.BCELoss()
    l1 = lambda v, r: torch.sum(torch.abs(v * r)) #L1 norm
    l2 = lambda v, r: torch.sqrt(torch.sum(torch.mul(v * r,v * r))) #L2 norm
    
    best_norm_weighted = np.inf
    best_pert_x = x_old
    
    k_i, loop_i, loop_change_class = orig_pred, 0, 0
    while loop_i < maxiters:
            
        zero_gradients(r)

        # Computing loss 
        loss_1 = bce(output, target)
        loss_2 = l2(v, r)
        loss = (loss_1 + lambda_ * loss_2)

        # Get the gradient
        loss.backward(retain_graph=True)
        grad_r = r.grad.data.cpu().numpy().copy()
        
        # Guide perturbation to the negative of the gradient
        ri = - grad_r
    
        # limit huge step
        ri *= alpha

        # Adds new perturbation to total perturbation
        r = r.clone().detach().cpu().numpy() + ri
        
        # For later computation
        r_norm_weighted = np.sum(np.abs(r * weights))
        
        # Ready to feed the model
        r = Variable(torch.FloatTensor(r), requires_grad=True) 
        
        # Compute adversarial example
        xprime = x_old + (1+overshoot) * r
        
        # Clip to stay in legitimate bounds
        xprime = clip(xprime, bounds[0], bounds[1])
        
        # Classify adversarial example
        output = model.forward(xprime)
        k_i = output.max(0, keepdim=True)[1].cpu().numpy()
        
        # Keep the best adverse at each iterations
        if k_i != orig_pred and r_norm_weighted < best_norm_weighted:
            best_norm_weighted = r_norm_weighted
            best_pert_x = xprime

        if k_i == orig_pred:
            loop_change_class += 1
            
        loop_i += 1 
        
    # Clip at the end no matter what
    best_pert_x = clip(best_pert_x, bounds[0], bounds[1])
    output = model.forward(best_pert_x)
    k_i = output.max(0, keepdim=True)[1].cpu().numpy()

    return orig_pred, k_i, best_pert_x.clone().detach().cpu().numpy(), loop_change_class 


def deepfool(x_old, net, max_iter, reduce_factor, bounds, weights=[], overshoot=0.002):
   
    input_shape = x_old.numpy().shape
    
    x = x_old.clone()
    x = Variable(x, requires_grad=True)
    
    output = net.forward(x)
    orig_pred = output.max(0, keepdim=True)[1] # get the index of the max log-probability

    origin = Variable(torch.tensor([orig_pred], requires_grad=False))

    print('\n')
    print(x_old)
    print(max_iter)
    print(reduce_factor)
    print(weights)
    print(bounds)
    I = []
    if orig_pred == 0:
        I = [0, 1]
    else:
        I = [1, 0]
        
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)
    
    k_i = origin
 
    loop_i = 0
    while torch.eq(k_i, origin) and loop_i < max_iter:

                
        # Origin class
        output[I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.numpy().copy()
        
        # Target class
        zero_gradients(x)
        output[I[1]].backward(retain_graph=True)
        cur_grad = x.grad.data.numpy().copy()
            
        # set new w and new f
        w = cur_grad - grad_orig
        f = (output[I[1]] - output[I[0]]).data.numpy()

        pert = abs(f)/np.linalg.norm(w.flatten())
    
        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)   
        
        if len(weights) > 0:
            r_i /= np.array(weights)

        # limit huge step
        r_i = reduce_factor * r_i / np.linalg.norm(r_i) 
            
        r_tot = np.float32(r_tot + r_i)
        
        
        pert_x = x_old + (1 + overshoot) * torch.from_numpy(r_tot)

        if len(bounds) > 0:
            pert_x = clip(pert_x, bounds[0], bounds[1])
                
        x = Variable(pert_x, requires_grad=True)
 
        output = net.forward(x)
        
        k_i = torch.tensor(np.argmax(output.data.cpu().numpy().flatten()))
                    
        loop_i += 1

    r_tot = (1+overshoot)*r_tot    
    pert_x = clip(pert_x, bounds[0], bounds[1])
    
 
    print(pert_x)
    print(loop_i)

    return orig_pred, k_i, pert_x.cpu(), loop_i