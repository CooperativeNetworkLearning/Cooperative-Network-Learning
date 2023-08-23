from data_loader import *

from model.gcn_link import GCNLinkPred
from model.gat_link import GATLinkPred
from model.sage_link import SAGELinkPred

def create_model(model_name, feature_dim, node_embedding_dim,hidden_size,num_heads):
    if model_name == "gcn":
        model = GCNLinkPred(feature_dim, hidden_size, node_embedding_dim)
    elif model_name == "gat":
        model = GATLinkPred(
            feature_dim, hidden_size, node_embedding_dim, num_heads
        )
    elif model_name == "sage":
        model = SAGELinkPred(feature_dim, hidden_size, node_embedding_dim)
    else:
        raise Exception("such model does not exist !")
    return model

