import torch

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

do_train   = True # default is true
max_length = 512
nos_labels = 7201 # total nos. of labels
batch_size = 6
num_workers= 4
embedding_batch_size = 32


roberta_model_path      = 'roberta-base'
roberta_embedding_dim   = 768
model_embedding_dim     = 100

learning_rate           = 1e-4
warmup_proportion       = 0.1
epochs                  = 4
momentum                = 0.9 #0.8
weight_decay            = 1e-1 #2e-4

"""
Hyperparameters Constant
"""
clusters_num            = 50
top_k                   = 7

"""
File path related constants
"""
model_save_path            = "outputs/model/model_state_dict_model.pt" # model_state_dict_model_4_epochs.pt" #model_state_dict_model_30_epochs.pt" #model_state_dict_model.pt"
prediction_save_path       = "outputs/predictions/final_outputs_targets.pickle"
metrics_path               = "outputs/metric/metric.txt"
document_embeddings_path   = "outputs/data_embeddings/data.embeddings"
kmeans_model_path          = "outputs/model/kmeans_prediction.pickle"


label_indices_map_path     = "data/label_indices_map_dict.pickle"
labels_embeddings          = "data/label_coocuurence_embeddings.pickle"
document_dataset_file_path = "data/document_dataset.pickle"


flat_file                  = "data/flat_{}_dataset_segments.pickle"
embed_file                 = "data/{}_segments_embedding.pickle"
