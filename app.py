import argparse

import pickle

import torch
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import model
import engine
import utils

def start():

    if config.do_train:

        print("Preparing the dataset")
        #train_dataset    = dataset.DocumentDataset('val', config.nos_labels)
        train_dataset    = dataset.DocumentDataset('train', config.nos_labels)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

        print("Preparing the model")
        the_model = model.DXML()
        the_model.to(config.device)


        param_optimizer = list(the_model.named_parameters())
        no_decay        = ['bias', "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
                {
                    "params": [
                        param for name, param in param_optimizer if not any(nd in name for nd in no_decay)
                        ],
                    "weight_decay": config.weight_decay
                    },
                {
                    "params": [
                        param for name, param in param_optimizer if any(nd in name for nd in no_decay)
                        ],
                    "weight_decay": 0.0
                    }
                ]
        optimizer       = AdamW(optimizer_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
        nos_train_steps = int( len(train_dataloader) * config.epochs)
        scheduler       = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = nos_train_steps*config.warmup_proportion, num_training_steps = nos_train_steps)


        for each_epoch in range(config.epochs):
            training_loss = engine.train(train_dataloader,  the_model, optimizer, scheduler)
            print(f'Epoch: {each_epoch}, Training Loss: {training_loss}\n')


        print('Saving the model')
        torch.save(the_model.state_dict(), config.model_save_path)

    
    the_model = model.DXML()
    print("Generating embedding and clustering them")
    print("\nLoading the model...")
    the_model.load_state_dict(torch.load(config.model_save_path, map_location='cpu')) # Map Location = 'cpu' is added to load the model with CPU incase.
    the_model.to(config.device)

    
    print("Preparing the dataset for embedding generation")
    train_dataset    = dataset.DocumentDataset('train', config.nos_labels)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False) # shuffle is mandotorily set to False
    engine.generate_clusters_FX(train_dataloader, the_model, config.nos_labels, True)

    
    """
    Evaluating the model
    """
    print("\nEvaluation started")
    test_dataset                 = dataset.DocumentDataset('test', config.nos_labels)
    test_dataloader              = torch.utils.data.DataLoader(test_dataset, batch_size = config.batch_size, shuffle=False)
    final_targets, final_outputs = engine.eval(test_dataloader, the_model)
    with open(config.prediction_save_path, 'wb') as handle:
        pickle.dump(final_targets, handle)
        pickle.dump(final_outputs, handle)


    print("\nNow starting evaluation...")
    utils.start_prediction_eval()



















def start_embedding_generation(dataset_split):

    embedding_dataset    = dataset.EmbeddingDataset(dataset_split)
    embedding_dataloader = torch.utils.data.DataLoader(embedding_dataset, batch_size = config.batch_size, shuffle=False)


    the_model            = model.DXML()
    print("\nLoading the model...")
    the_model.load_state_dict(torch.load(config.model_save_path, map_location='cpu')) # Map Location = 'cpu' is added to load the model with CPU incase.
    the_model.to(config.device)

    """
    Generating the embeddings for the segments
    """
    engine.generate_segment_embeddings(embedding_dataloader, the_model, dataset_split)







if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--do-Train', dest="doTrain", required=True, type=str, help="True or False")
    parser.add_argument('--do-Generate', dest="doGenerate", required=True, type=str, help="True or False")
    args = parser.parse_args()

    if args.doTrain.lower() == "true":
        config.do_train = True
    else:
        config.do_train = False

    if args.doGenerate.lower() == "true":
        start_embedding_generation("train")
    else:
        start() # start the app
