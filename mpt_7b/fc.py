from torch import nn
FC_CLASS_REGISTRY = {'torch': nn.Linear}
try:
    import transformer_engine.pytorch as te
    FC_CLASS_REGISTRY['te'] = te.Linear
except:
    pass