from .densenet import DenseNet
import torch

def getModel(training=False,model_to_use='densenet',**kwargs):

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    
    model = DenseNet(growth_rate=12, depth=100, reduction=0.5,**kwargs)
    
    model = torch.nn.DataParallel(model)
    model.eval()
    
    '''
    
    if training :
        model.train()
        model = torch.nn.DataParallel(model)
        print("The model is in training mode")
    '''

    print("No of params in model is " , sum(p.numel() for p in model.parameters() if p.requires_grad))
    model  = model.to(device)
    print(f"model is loaded on GPU {next(model.parameters()).is_cuda}")
    return model
