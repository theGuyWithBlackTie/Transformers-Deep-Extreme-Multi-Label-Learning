import config

import torch
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
import pickle

import math
import os

def loss_fn(output, target):
    return F.smooth_l1_loss(output, target)


def train(data_loader, model, optimizer, scheduler):
    model.train()
    total_loss = 0

    for bi, data in tqdm(enumerate(data_loader), total=len(data_loader)):

        ids            = data["ids"]
        mask           = data["mask"]
        token_type_ids = data["token_type_ids"]
        targets        = data["target_embeds"]      # This is target embedding

        ids            = ids.to(config.device, dtype=torch.long)
        mask           = mask.to(config.device, dtype=torch.long)
        token_type_ids = token_type_ids.to(config.device, dtype=torch.long)
        targets        = targets.to(config.device, dtype=torch.float)

        optimizer.zero_grad()
        output         = model(ids, mask, token_type_ids)
        loss           = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss    += loss.item()

    return total_loss/len(data_loader)



"""
Function to generate the embeddings of the dataset
"""
def generate_clusters_FX(dataloader, model, total_nos_labels, is_generate_clusters=False):
    if is_generate_clusters == False and os.path.isfile(config.kmeans_model_path):
        print('Not generating the clusters')
        return

    model.eval()
    FX      = np.zeros((1,100)) # embedding size is 100
    targets = np.zeros((1,total_nos_labels))
    print('\nGenerating FX...')

    with torch.no_grad():

        for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):

            ids            = data["ids"]
            mask           = data["mask"]
            token_type_ids = data["token_type_ids"]
            target         = data["targets"]        # This is 1-hot label row
            target         = target.squeeze(1)

            ids            = ids.to(config.device, dtype=torch.long)
            mask           = mask.to(config.device, dtype=torch.long)
            token_type_ids = token_type_ids.to(config.device, dtype=torch.long)

            output         = model(ids, mask, token_type_ids)

            FX             = np.append(FX, output.cpu().detach().numpy(), 0)
            targets        = np.append(targets, target.cpu().detach().numpy(),0)



    """
    Removing first dummy row of zeros
    """
    FX      = FX[1:,:]   
    targets = targets[1:,:]
    print('FX.shape: ',FX.shape)

    kMeans_model = generate_clusters(FX)


    """
    Saving the KMeans model"
    """
    with open(config.kmeans_model_path, 'wb') as pickle_file:
        pickle.dump(kMeans_model, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(FX, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(targets, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


    print('\nKMeans Model, FX and targets are saved at ',config.kmeans_model_path)




def generate_clusters(FX):
    print('\nStarting clustering...')
    kMeans_model = KMeans(n_clusters = config.clusters_num, max_iter=1000, random_state=42)
    kMeans_model.fit(FX)
    return kMeans_model



"""
Function to evaluate the model
"""
def eval(dataloader, model):
    model.eval()
    print('Loading the KMeans prediction model...')
    final_outputs = []
    final_targets = []
    loss          = 0

    with open(config.kmeans_model_path, 'rb') as pickle_file:
        kMeans_model = pickle.load(pickle_file)
        FX           = pickle.load(pickle_file)
        targets      = pickle.load(pickle_file)

    with open(config.labels_embeddings, 'rb') as handle:
        label_embeddings = pickle.load(handle).detach().numpy()


    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):

            ids            = data["ids"]
            mask           = data["mask"]
            token_type_ids = data["token_type_ids"]
            target_embed   = data["target_embeds"]
            test_target    = data["targets"]


            ids            = ids.to(config.device, dtype=torch.long)
            mask           = mask.to(config.device, dtype=torch.long)
            token_type_ids = token_type_ids.to(config.device, dtype=torch.long)
            target_embed   = target_embed.to(config.device, dtype=torch.long)
            test_target    = test_target.to(config.device, dtype=torch.long)


            outputs        = model(ids, mask, token_type_ids)
            loss          += loss_fn(outputs, target_embed)

            outputs        = outputs.cpu().detach().numpy()

            """
            Finding the nearest cluster
            """
            for each_row in outputs: # because outputs is of batch_size
                # Find the closest cluster
                cluster_no      = find_ZI_star(each_row, kMeans_model)

                # Find all the elements in the closest cluster found
                cluster_indices = ClusterIndicesNumpy(cluster_no, kMeans_model.labels_)

                # Following 'find_k_nn' just returns all the closest cluster's element with their distances from the 'each_row'
                k_nn_elems      = find_k_nn(cluster_indices, each_row, FX)
                # This actually takes top_K elements based on distance and takes its ground truth
                y_cap           = generate_y_cap(k_nn_elems, targets)

                #y_cap           = generate_y_cap_based_on_embed_distance(k_nn_elems, targets, each_row, label_embeddings) # Predicting and sorting the labels based on close
                print('Y_cap -> ',y_cap)
                final_outputs.append(y_cap)

            final_targets.extend(test_target.cpu().detach().numpy().tolist())


    print('Eval Loss: ',loss/len(dataloader))
    return final_targets, final_outputs




"""
Find close clusters
"""
def find_ZI_star(vector, kMeans_model):
    all_centroids = kMeans_model.cluster_centers_
    distance      = math.inf
    cluster_no    = math.inf

    for each_elem in range(0, len(all_centroids)):
        eachCentroid = all_centroids[each_elem]
        temp_dist    = np.linalg.norm(vector-eachCentroid)
        if temp_dist < distance:
            cluster_no = each_elem
            distance   = temp_dist
    return cluster_no


"""
This will return all the element from FX which belongs to clusNum
"""
def ClusterIndicesNumpy(clustNum, labels_array):
    return np.where(labels_array == clustNum)[0]


"""
From a subset of FX, find nearest K neighbours
"""
def find_k_nn(cluster_indices, vector, FX):
    elem_dist_dict = {}
    for eachElem in cluster_indices:
        each_cluster_elem = FX[eachElem]
        distance          = np.linalg.norm(vector-each_cluster_elem)
        elem_dist_dict[eachElem] = distance


    # Sorting the dict based on distance - value
    sorted_elem_dist_dict = {}
    sorted_keys = sorted(elem_dist_dict, key=elem_dist_dict.get, reverse=False)
    for i in sorted_keys:
        sorted_elem_dist_dict[i] = elem_dist_dict[i]
    return sorted_elem_dist_dict



def generate_y_cap(k_nn_elems, targets):
    top_K = config.top_k
    
    top_K_indices = list(k_nn_elems.keys())[0:top_K]
    all_labels    = []
    for eachElem in top_K_indices:
        row = targets[eachElem]
        all_labels.extend(np.where(row == 1)[0].tolist())
    all_labels = sorted(sorted(all_labels, reverse=True), key=all_labels.count, reverse=True)
    temp     = set()
    temp_add = temp.add
    y_cap    = [x for x in all_labels if not (x in temp or temp_add(x))]
    return y_cap











"""
Following section is related to generating the embeddings of the segments
"""

def generate_segment_embeddings(data_loader, model, dataset_split):

    segments_embedding_file = config.embed_file.format(dataset_split)
    model.eval()

    with torch.no_grad():

        segments_embeds = []
        orig_segments   = []

        for index, data in tqdm( enumerate(data_loader), total = len(data_loader)):

            ids            = data["ids"].to(config.device)
            mask           = data["mask"].to(config.device)
            token_type_ids = data["token_type_ids"].to(config.device)

            segments       = data["original_segment"]

            embedding      = model(ids, mask, token_type_ids)

            segments_embeds.append(embedding.cpu().detach())
            orig_segments.append(segments)



        # Saving the segment embeddings in batches though. Total len would be equal to total_segments / batch_size
        with open(segments_embedding_file, 'wb') as handle:
            pickle.dump(segments_embeds, handle)
            pickle.dump(orig_segments, handle)

    print("Segment embeddings are saved at: {}".format(segments_embedding_file))





'''
Following function predict labels for the sentence segments
'''
def predict_segment_labels(data_loader, model):

    print("Loading the KMeans models")
    with open(config.kmeans_model_path, 'rb') as pickle_file:
        kMeans_model = pickle.load(pickle_file)
        FX           = pickle.load(pickle_file)
        targets      = pickle.load(pickle_file)


    model.eval()

    with torch.no_grad():

        for index, data in tqdm( enumerate(data_loader), total = len(data_loader)):

            ids            = data["ids"].to(config.device)
            mask           = data["mask"].to(config.device)
            token_type_ids = data["token_type_ids"].to(config.device)
            target_labels  = data["target_labels"]



            outputs        = model(ids, mask, token_type_ids)
            outputs        = outputs.cpu().detach().numpy()


            for each_row in outputs: # because outputs is of batch_size
                # Find the closest cluster
                cluster_no      = find_ZI_star(each_row, kMeans_model)

                # Find all the elements in the closest cluster found
                cluster_indices = ClusterIndicesNumpy(cluster_no, kMeans_model.labels_)

                # Following 'find_k_nn' just returns all the closest cluster's element with their distances from the 'each_row'
                k_nn_elems      = find_k_nn(cluster_indices, each_row, FX)
                # This actually takes top_K elements based on distance and takes its ground truth
                y_cap           = generate_y_cap(k_nn_elems, targets)
                y_cap           = y_cap[:3]
                final_outputs.append(y_cap)


    
"""
Following function selects the labels for sentence segment.
It takes all the elements in the closest cluster, and selects one label based on how close label embeddings are to the segment embedding
"""
def generate_y_cap_based_on_embed_distance(k_nn_elems, targets, doc_embedding, label_embeddings):
    top_K = config.top_k

    top_k_indices = list(k_nn_elems.keys())[0:top_K]
    all_labels    = []

    for each_elem in top_k_indices:
        row        = targets[each_elem]
        all_labels.extend(np.where(row == 1)[0].tolist()) # Taking all the labels indices which are 1 and then appending it

    all_labels    = set(all_labels)
    sorted_labels = sort_labels_based_on_distance(all_labels,doc_embedding, label_embeddings)
    return sorted_labels



"""
Following function sorts the labels based on their distance between docment embedding and label_embedding
"""
def sort_labels_based_on_distance(labels, doc_embedding, label_embeddings):
    labels_distance = {}
    for each_elem in labels:
        labels_distance[each_elem] = np.linalg.norm(label_embeddings[each_elem]-doc_embedding)

    sorted_labels = sorted(labels_distance, key=labels_distance.get, reverse=False) # We need ascending order
    return sorted_labels
