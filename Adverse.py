# Misc
import numpy as np

# Pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

# Clipping function
def clip(current, low_bound, up_bound):
    assert(len(current) == len(up_bound) and len(low_bound) == len(up_bound))
    low_bound = torch.FloatTensor(low_bound)
    up_bound = torch.FloatTensor(up_bound)
    clipped = torch.max(torch.min(current, up_bound), low_bound)
    return clipped


def lowProFool(x, model, weights, bounds, maxiters, alpha, lambda_):
    """
    Generates an adversarial examples x' from an original sample x

    :param x: tabular sample
    :param model: neural network
    :param weights: feature importance vector associated with the dataset at hand
    :param bounds: bounds of the datasets with respect to each feature
    :param maxiters: maximum number of iterations ran to generate the adversarial examples
    :param alpha: scaling factor used to control the growth of the perturbation
    :param lambda_: trade off factor between fooling the classifier and generating imperceptible adversarial example
    :return: original label prediction, final label prediction, adversarial examples x', iteration at which the class changed
    """

    r = Variable(torch.FloatTensor(1e-4 * np.ones(x.numpy().shape)), requires_grad=True) 
    v = torch.FloatTensor(np.array(weights))
    
    output = model.forward(x + r)
    orig_pred = output.max(0, keepdim=True)[1].cpu().numpy()
    target_pred = np.abs(1 - orig_pred)
    
    target = [0., 1.] if target_pred == 1 else [1., 0.]
    target = Variable(torch.tensor(target, requires_grad=False)) 
    
    lambda_ = torch.tensor([lambda_])
    
    bce = nn.BCELoss()
    l1 = lambda v, r: torch.sum(torch.abs(v * r)) #L1 norm
    l2 = lambda v, r: torch.sqrt(torch.sum(torch.mul(v * r,v * r))) #L2 norm
    
    best_norm_weighted = np.inf
    best_pert_x = x
    
    loop_i, loop_change_class = 0, 0
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
        xprime = x + r
        
        # Clip to stay in legitimate bounds
        xprime = clip(xprime, bounds[0], bounds[1])
        
        # Classify adversarial example
        output = model.forward(xprime)
        output_pred = output.max(0, keepdim=True)[1].cpu().numpy()
        
        # Keep the best adverse at each iterations
        if output_pred != orig_pred and r_norm_weighted < best_norm_weighted:
            best_norm_weighted = r_norm_weighted
            best_pert_x = xprime

        if output_pred == orig_pred:
            loop_change_class += 1
            
        loop_i += 1 
        
    # Clip at the end no matter what
    best_pert_x = clip(best_pert_x, bounds[0], bounds[1])
    output = model.forward(best_pert_x)
    output_pred = output.max(0, keepdim=True)[1].cpu().numpy()

    return orig_pred, output_pred, best_pert_x.clone().detach().cpu().numpy(), loop_change_class 

# Forked from https://github.com/LTS4/DeepFool
def deepfool(x_old, net, maxiters, alpha, bounds, weights=[], overshoot=0.002):
    """
    :param image: tabular sample
    :param net: network 
    :param maxiters: maximum number of iterations ran to generate the adversarial examples
    :param alpha: scaling factor used to control the growth of the perturbation
    :param bounds: bounds of the datasets with respect to each feature
    :param weights: feature importance vector associated with the dataset at hand
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    
    input_shape = x_old.numpy().shape
    x = x_old.clone()
    x = Variable(x, requires_grad=True)
    
    output = net.forward(x)
    orig_pred = output.max(0, keepdim=True)[1] # get the index of the max log-probability

    origin = Variable(torch.tensor([orig_pred], requires_grad=False))

    I = []
    if orig_pred == 0:
        I = [0, 1]
    else:
        I = [1, 0]
        
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)
    
    k_i = origin
 
    loop_i = 0
    while torch.eq(k_i, origin) and loop_i < maxiters:
                
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
        r_i = alpha * r_i / np.linalg.norm(r_i) 
            
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

    return orig_pred, k_i, pert_x.cpu(), loop_i