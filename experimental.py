#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    ANFIS in torch: some simple functions to supply data and plot results.
    @author: James Power <james.power@mu.ie> Apr 12 18:13:10 2019
"""

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

dtype = torch.float


class TwoLayerNet(torch.nn.Module):
    '''
        From the pytorch examples, a simjple 2-layer neural net.
        https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    '''
    def __init__(self, d_in, hidden_size, d_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(d_in, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, d_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


def linear_model(x, y, epochs=200, hidden_size=10):
    '''
        Predict y from x using a simple linear model with one hidden layer.
        https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
    '''
    assert x.shape[0] == y.shape[0], 'x and y have different batch sizes'
    d_in = x.shape[1]
    d_out = y.shape[1]
    model = TwoLayerNet(d_in, hidden_size, d_out)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    errors = []
    for t in range(epochs):
        y_pred = model(x)
        tot_loss = criterion(y_pred, y)
        perc_loss = 100. * torch.sqrt(tot_loss).item() / y.sum()
        errors.append(perc_loss)
        if t % 10 == 0 or epochs < 20:
            print('epoch {:4d}: {:.5f} {:.2f}%'.format(t, tot_loss, perc_loss))
        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()
    return model, errors


def plotErrors(errors):
    '''
        Plot the given list of error rates against no. of epochs
    '''
    plt.figure(figsize=(8, 4.5))

    plt.rcParams['font.sans-serif'] = 'Arial'
    
    plt.plot(range(len(errors)), errors, '-ro', label='errors')
    plt.ylabel('Loss Function', fontsize=12, color='black')
    plt.xlabel('Epoch', fontsize=12, color='black')
    
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    
    plt.tight_layout()
    plt.savefig('errors_plot.png', transparent=True)
    plt.show()

def plotResults(y_actual, y_predicted):
    '''
        Plot the actual and predicted y values (in different colours).
    '''
    plt.plot(range(len(y_predicted)), y_predicted.detach().numpy(),
             'r', label='trained')
    plt.plot(range(len(y_actual)), y_actual.numpy(), 'b', label='original')
    plt.legend(loc='upper left')
    plt.show()


def _plot_mfs(var_name, fv, x):
    '''
        A simple utility function to plot the MFs for a variable.
        Supply the variable name, MFs and a set of x values to plot.
    '''
    # Sort x so we only plot each x-value once:
    xsort, _ = x.sort()
    
    plt.figure(figsize=(8, 4.5))

    plt.rcParams['font.sans-serif'] = 'Arial'
    
    for mfname, yvals in fv.fuzzify(xsort):
        plt.plot(xsort.tolist(), yvals.tolist(), label=mfname)
        
    
    plt.xlabel('Scaled Values for variable {}'.format(var_name))
    plt.ylabel('Membership Values')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    #plt.legend(bbox_to_anchor=(1., 0.95))
    plt.savefig('{}_memberships.png'.format(var_name), transparent=True)
    plt.show()


def plot_all_mfs(model, x):
    for i, (var_name, fv) in enumerate(model.layer.fuzzify.varmfs.items()):
        _plot_mfs(var_name, fv, x[:, i])


def calc_error(y_pred, y_actual):
    with torch.no_grad():
        tot_loss = F.mse_loss(y_pred, y_actual)
        rmse = torch.sqrt(tot_loss).item()
        perc_loss = torch.mean(100. * torch.abs((y_pred - y_actual)
                               / y_actual))
    return(tot_loss, rmse, perc_loss)


def test_anfis(model, data, show_plots=False):
    '''
        Do a single forward pass with x and compare with y_actual.
    '''
    x, y_actual = data.dataset.tensors
    if show_plots:
        plot_all_mfs(model, x)
    print('### Testing for {} cases'.format(x.shape[0]))
    y_pred = model(x)
    mse, rmse, perc_loss = calc_error(y_pred, y_actual)
    print('MS error={:.5f}, RMS error={:.5f}, percentage={:.2f}%'
          .format(mse, rmse, perc_loss))
    if show_plots:
        plotResults(y_actual, y_pred)

        
def classification_predictions(model, loader):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    with torch.no_grad():  # No need to track gradients
        for x in loader:
            logits = model(x)
            y_probs = torch.softmax(logits, dim=1)
            y_pred = torch.argmax(y_probs, dim=1) - 1  # I subtract 1 to labels to match my notation [-1, 0, 1]
            predictions.extend(y_pred.tolist())
    return predictions        
        
def test_anfis_classifier(model, data, show_plots=False):
    '''
        Addition by Gian to work with AnfisNetClassifier
        Do a single forward pass with x and compare with y_actual.
    '''
    x, y_actual = data.dataset.tensors
    if show_plots:
        plot_all_mfs(model, x)
    print('### Testing for {} cases'.format(x.shape[0]))
    y_pred = model(x)
    mse, rmse, perc_loss = calc_error(y_pred, y_actual)
    print('MS error={:.5f}, RMS error={:.5f}, percentage={:.2f}%'
          .format(mse, rmse, perc_loss))
    if show_plots:
        plotResults(y_actual, y_pred)


def train_anfis_with(model, data, optimizer, criterion,
                     epochs=500, show_plots=False):
    '''
        Train the given model using the given (x,y) data.
    '''
    errors = []  # Keep a list of these for plotting afterwards
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print('### Training for {} epochs, training size = {} cases'.
          format(epochs, data.dataset.tensors[0].shape[0]))
    for t in range(epochs):
        # Process each mini-batch in turn:
        for x, y_actual in data:
            y_pred = model(x)
            # Compute and print loss
            loss = criterion(y_pred, y_actual)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("NaN or Inf in loss detected")
                break
            
            #for name, param in model.named_parameters():
            #    if param.grad is not None:
            #        print(f"{name} gradient: {param.grad.norm()}")
            
        # Epoch ending, so now fit the coefficients based on all data:
        x, y_actual = data.dataset.tensors
        with torch.no_grad():
            model.fit_coeff(x, y_actual)
        # Get the error rate for the whole batch:
        y_pred = model(x)
        mse, rmse, perc_loss = calc_error(y_pred, y_actual)
        errors.append(perc_loss)
        # Print some progress information as the net is trained:
        if epochs < 30 or t % 10 == 0:
            print('epoch {:4d}: MSE={:.5f}, RMSE={:.5f} ={:.2f}%'
                  .format(t, mse, rmse, perc_loss))
    # End of training, so graph the results:
    if show_plots:
        plotErrors(errors)
        y_actual = data.dataset.tensors[1]
        y_pred = model(data.dataset.tensors[0])
        plotResults(y_actual, y_pred)

def train_anfis_classifier_with(model, data, optimizer, epochs=500, show_plots=False):
    '''
        Train the classifier ANFIS model using the given (x,y) data.
        Modified version by Gian in order to use misclassification error, or 
        torch.nn.CrossEntropyLoss()
        The input for crossentropy should be logits for each class outputted by model.
        Target should be class labels in the range [0,1,...,num_classes-1].
        'optimizer' must be from torch, e.g.
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        Note: 'data' is a pytorch dataloader with (inputs, targets)
    '''
    criterion = torch.nn.CrossEntropyLoss()
    errors = []  # Keep a list of these for plotting afterwards
    print('### Training for {} epochs, training size = {} cases'.format(epochs, len(data.dataset)))
    for t in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for x, y_actual in data:
            logits = model(x)  # Forward pass, model outputs logits for CrossEntropyLoss
            assert not torch.isnan(logits).any(), "NaN found in logits"
            optimizer.zero_grad()  # Zero the parameter gradients
            # print("Logits shape:", logits.shape)  # Should be [64, num_classes]
            loss = criterion(logits, y_actual)  # Compute loss
            # print('Loss={:.3f}'.format(loss))
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("NaN or Inf in loss detected")
                break
            loss.backward()  # Backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            '''
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name} gradient: {param.grad.norm()}")
            '''    
            optimizer.step()  # Optimize
            running_loss += loss.item() * x.size(0)  # Multiply by batch size
        epoch_loss = running_loss / len(data.dataset)  # Average loss for the epoch
        errors.append(epoch_loss)  # Append to errors list for plotting
        
        # Print some progress information as the net is trained:
        if t % 10 == 0 or t == epochs - 1:  # Every 10 epochs or last epochw4
            print('epoch {:4d}: Loss={:.5f}'.format(t, epoch_loss))
        
    # End of training
    if show_plots: 
        plotErrors(errors)  # Function to plot the training error
        

def train_anfis(model, data, epochs=500, show_plots=False):
    '''
        Train the given model using the given (x,y) data.
    '''
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.99)
    criterion = torch.nn.MSELoss(reduction='sum')
    train_anfis_with(model, data, optimizer, criterion, epochs, show_plots)


if __name__ == '__main__':
    x = torch.arange(1, 100, dtype=dtype).unsqueeze(1)
    y = torch.pow(x, 3)
    model, errors = linear_model(x, y, 100)
    plotErrors(errors)
    plotResults(y, model(x))
