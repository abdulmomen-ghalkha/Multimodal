import torch.nn as nn
import torchvision.transforms as transf
import pandas as pd
from skimage import io
from torch.utils.data import DataLoader, Dataset
import networkx as nx
import random
import torch
import numpy as np



############### Create data sample list #################
def create_data_sample(root, shuffle=False, nat_sort=False):
    f = pd.read_csv(root)
    data_samples = []
    pred_val = []
    for idx, row in f.iterrows():
        img_paths = row.values[5:7]
        features = row.values[1:5]
        data_samples.append([features, img_paths])
    return data_samples
#############################################################

class DataFeed_image_pos(Dataset):
    '''
    A class retrieving a tuple of (image,label). It can handle the case
    of empty classes (empty folders).
    '''
    def __init__(self,root_dir, nat_sort = False, transform=None, init_shuflle = True):
        self.root = root_dir
        self.samples = create_data_sample(self.root,shuffle=init_shuflle,nat_sort=nat_sort)
        self.transform = transform


    def __len__(self):
        return len( self.samples )

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pos = sample[0].astype(np.float32)
        img = io.imread(sample[1][0])
        img = self.transform(img)
        label = sample[1][1]
        #if "pos_height" in self.modalities and "images" in self.modalities:
        #    return (pos, img, label)
        #elif "pos_height" in self.modalities:
        #    return (pos,label)
        #else:
        #    return (img,label)
        return ({"pos_height": pos, "images": img}, label)
    




def read_drone_6G(distribution="IID", batchsize=64, no_users=20):
    if distribution == "IID":
        dataset_dir = "datasets/drone/feature_IID/"
    elif distribution == "NIID":
        dataset_dir = "datasets/drone/feature_NIID/"
    else:
        raise ValueError("Invalid dataset distribution.")
    
    
    no_users = no_users
    batch_size = batchsize
    no_classes = 64
    available_modalities = ["pos_height", "images"]
    modality_size = {"pos_height": 128, "images": 128}
    group_definitions = {
        1: ["pos_height"],        # Group 1: Only pos_height
        2: ["images"],            # Group 2: Only images
        3: ["pos_height", "images"]  # Group 3: Both modalities
    }
    distribution_weights = [0.2, 0.3, 0.5]
    # Generate user_groups with weighted random choices
    user_groups = random.choices([1, 2, 3], weights=distribution_weights, k=no_users)

    # Assign modalities to users based on their group
    user_modalities = [group_definitions[group] for group in user_groups]

    img_resize = transf.Resize((224, 224))
    img_norm = transf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    proc_pipe = transf.Compose(
        [transf.ToPILImage(),
        img_resize,
        transf.ToTensor(),
        img_norm]
    )

    train_loaders = []
    test_loaders = []
    val_loaders = []

    for user_id in range(no_users):
        train_dir = dataset_dir + f'user_{user_id}_pos_height_beam_train.csv'
        val_dir = dataset_dir + f'user_{user_id}_pos_height_beam_val.csv'
        test_dir = dataset_dir + f'user_{user_id}_pos_height_beam_test.csv'
        
        train_dataset = DataFeed_image_pos(train_dir, transform=proc_pipe)
        val_dataset = DataFeed_image_pos(root_dir=val_dir, transform=proc_pipe)
        test_dataset = DataFeed_image_pos(root_dir=test_dir, transform=proc_pipe)
        
        
        train_loaders.append(DataLoader(train_dataset,
                                batch_size=batch_size,
                                #num_workers=8,
                                shuffle=True))
        val_loaders.append(DataLoader(val_dataset,
                                batch_size=batch_size,
                                #num_workers=8,
                                shuffle=False))
        test_loaders.append(DataLoader(test_dataset,
                                batch_size=batch_size,
                                #num_workers=8,
                                shuffle=False))
        

    return train_loaders, val_loaders, test_loaders, user_modalities, available_modalities, user_groups, group_definitions



def cross_entropy_loss_with_l2(logits, targets, l2_strength=0):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)
    #l2_reg = sum(param.pow(2).sum() for param in model.parameters())
    #loss + l2_strength * l2_reg
    return loss 
    

def construct_mixing_matrix(Adj, method="metropolis"):
    n = Adj.shape[0]
    W = np.zeros((n, n))  # Initialize weight matrix

    for i in range(n):
        degree_i = np.sum(Adj[i, :])

        for j in range(n):
            if Adj[i, j] == 1.0:
                degree_j = np.sum(Adj[j, :])
    
                if method == "metropolis":
                    W[i, j] = 1 / (max(degree_i, degree_j) + 1)
                elif method == "uniform":
                    W[i, j] = 1 / degree_i

        # Diagonal weight
        W[i, i] = 1 - np.sum(W[i, :])

    return W


def construct_mixing_matrix(Adj, method="metropolis"):
    n = Adj.shape[0]
    W = np.zeros((n, n))  # Initialize weight matrix

    for i in range(n):
        degree_i = np.sum(Adj[i, :])

        for j in range(n):
            if Adj[i, j] == 1.0:
                degree_j = np.sum(Adj[j, :])
    
                if method == "metropolis":
                    W[i, j] = 1 / (max(degree_i, degree_j) + 1)
                elif method == "uniform":
                    W[i, j] = 1 / degree_i

        # Diagonal weight
        W[i, i] = 1 - np.sum(W[i, :])

    return W

def create_random_topology(num_users, similarity_matrix, edge_probability=0.3):
    """
    Creates a connected random topology using NetworkX.
    Returns the adjacency matrix.
    """
    while True:
        graph = nx.erdos_renyi_graph(num_users, edge_probability)
        adjacency_matrix = nx.to_numpy_array(graph)
        new_adj = np.multiply(adjacency_matrix, similarity_matrix)
        new_graph = nx.from_numpy_array(new_adj)
        if nx.is_connected(new_graph):
            break

    # Convert graph to adjacency matrix
    adjacency_matrix = nx.to_numpy_array(new_graph)
    return adjacency_matrix



def generate_random_adj_mixing_matrices(available_modalities, user_modalities, no_users):
    similarity_matrix = np.zeros((no_users, no_users), dtype=int)

    # Construct the adjacency matrix
    for i in range(no_users):
        for j in range(no_users):
            if i != j:  # No self-loops
                # Check if users i and j share any modalities
                if set(user_modalities[i]) & set(user_modalities[j]):
                    similarity_matrix[i, j] = 1

    adjacency_matrix = create_random_topology(20, similarity_matrix, edge_probability=0.3)
    G = nx.from_numpy_array(adjacency_matrix)
    # Similarity matrices
    adj_per_modality = {}
    for modality in available_modalities:
        adj = np.zeros((no_users, no_users))
        for node in range(no_users):
            for neighbor in G.neighbors(node):
                if modality in user_modalities[neighbor] and modality in user_modalities[node]:
                    adj[node, neighbor] = 1.    
        adj_per_modality[modality] = adj
    
    mixing_matrices = {}
    for modality in available_modalities:
        mixing_matrices[modality] = construct_mixing_matrix(adj_per_modality[modality], method="metropolis")

    return adjacency_matrix, mixing_matrices
    


    


    


def validate_user_models(user_id, model, val_loader, criterion, local_modalities, device):
    """
    Validates a trained multi-modal model.

    Args:
        user_id (int): User identifier.
        model (Classifier): Multi-modal classification model.
        val_loader (DataLoader): Validation data loader.
        criterion (nn.CrossEntropyLoss): Loss function.
        local_modalities (list): Modalities to use (e.g., ['image', 'pos']).
        device (torch.device): Device (CPU/GPU).

    Returns:
        dict: Validation loss and accuracy.
    """
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch

            # Prepare input data for selected modalities
            modality_inputs = {mod: inputs[mod].to(device) for mod in local_modalities}
            labels = labels.to(device)

            # Forward pass
            outputs = model(**modality_inputs)
            loss = criterion(outputs, labels)

            # Accumulate metrics
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    # Compute loss and accuracy
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    print(f"User {user_id + 1} - Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    return {"loss": avg_loss, "accuracy": accuracy}


def fmtl_aggregation(classifiers, optimizers, train_loaders, user_modalities, user_groups, no_users, adjacency_matrix, P, alpha, lambda_reg, eta, criterion, device):
    """
    Perform FMTL Aggregation while only updating model.classifier.
    
    Args:
        classifiers (dict): Dictionary of user classifier models.
        optimizers (dict): Dictionary of user optimizers.
        train_loaders (dict): Dictionary of user train loaders.
        user_modalities (dict): Dictionary mapping each user to their available modalities.
        user_groups (dict): Mapping of users to groups.
        no_users (int): Number of users.
        adjacency_matrix (torch.Tensor): Adjacency matrix for FMTL.
        P (dict): Projection matrices for FMTL.
        alpha (float): Learning rate scaling factor.
        lambda_reg (float): Regularization parameter.
        eta (float): Step size for projection updates.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Computation device.
    """

    for user_id in range(no_users):
        client_model = classifiers[user_id]  # Get user's model
        optimizer = optimizers[user_id]      # Get optimizer
        train_loader = train_loaders[user_id]  # Get train loader
        modalities = user_modalities[user_id]  # Get list of modalities for user
        group = user_groups[user_id]           # Get user's group

        # freezing first layers 
        for mod in modalities:
            for param in client_model.sub_networks[mod].parameters():
                param.requires_grad = False  # Freezes the feature extractor

        epoch_train_loss = 0.0
        correct_train_predictions = 0
        total_train_samples = 0

        # Set model to training mode
        client_model.train()

        for batch in train_loader:
            inputs, labels = batch

            # Prepare input data for selected modalities
            modality_inputs = {mod: inputs[mod].to(device) for mod in modalities}
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass using selected modalities
            predictions = client_model(**modality_inputs)
            loss = criterion(predictions, labels)

            # Backpropagation (only for classifier)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            _, predicted = torch.max(predictions, dim=1)
            correct_train_predictions += (predicted == labels).sum().item()
            total_train_samples += labels.size(0)

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_accuracy = correct_train_predictions / total_train_samples

        # ==============================
        #        FMTL Parameter Update
        # ==============================
        with torch.no_grad():
            # Extract classifier parameters only
            theta_i = torch.cat([param.view(-1) for param in client_model.classifier.parameters()])
            sum_P_terms = torch.zeros_like(theta_i, device=device)
            projection_norm = 0

            for j in range(no_users):
                if adjacency_matrix[user_id, j] == 1:
                    P_ij = P[(user_id, j)]
                    P_ji = P[(j, user_id)]
                    theta_j = torch.cat([param.view(-1) for param in classifiers[j].classifier.parameters()])
                    
                    sum_P_terms += P_ij.T @ (P_ij @ theta_i - P_ji @ theta_j)
                    projection_norm += torch.linalg.norm(P_ij @ theta_i - P_ji @ theta_j, ord=1, dim=0)

    

            # Apply FMTL update rule
            theta_i -= alpha * lambda_reg * sum_P_terms

            # Update classifier parameters only
            idx = 0
            for param in client_model.classifier.parameters():
                numel = param.numel()
                param.data.copy_(theta_i[idx:idx+numel].reshape(param.size()))
                idx += numel

            # Projection matrix update
            for j in range(no_users):
                if adjacency_matrix[user_id, j] == 1:
                    P_ij = P[(user_id, j)]
                    P_ji = P[(j, user_id)]
                    theta_j = torch.cat([param.view(-1) for param in classifiers[j].classifier.parameters()])
                    
                    P[(user_id, j)] -= eta * lambda_reg * torch.outer(P_ij @ theta_i - P_ji @ theta_j, theta_i)



# Decentralized aggregation function
def per_modelaity_decentralized_aggregation(user_models, mixing_matrices, available_modalities, user_modalities):
    num_users = len(user_models)
    for modality in available_modalities:
        # Get the mixing matrix for the current modality
        mixing_matrix = mixing_matrices[modality]
        
        # Convert user model parameters to vectors for aggregation
        aggregated_models = []
        aggregated_updates = []
        for user_id, user_model in enumerate(user_models):
            if modality in user_modalities[user_id]:
                aggregated_models.append(torch.nn.utils.parameters_to_vector(user_model.sub_networks[modality].parameters()))
                aggregated_updates.append(torch.zeros_like(aggregated_models[-1]))
            else:
                aggregated_models.append(0)
                aggregated_updates.append(0)
        
        
        # Perform model aggregation based on the mixing matrix for this modality
        for i in range(num_users):
            for j in range(num_users):
                if mixing_matrix[i, j] > 0:
                    aggregated_updates[i] += mixing_matrix[i, j] * aggregated_models[j]
        
        # Update user models with aggregated parameters for the current modality
        for user_id in range(num_users):
            if modality in user_modalities[user_id]:
                torch.nn.utils.vector_to_parameters(aggregated_updates[user_id], user_models[user_id].sub_networks[modality].parameters())



def train_local_model(local_modalities, model, train_loader, criterion, optimizer, epochs, device):
    """
    Trains a local multi-modal model.

    Args:
        local_modalities (list): Modalities to use (e.g., ['image', 'pos']).
        model (Classifier): Multi-modal classification model.
        train_loader (DataLoader): Training data loader.
        criterion (nn.CrossEntropyLoss): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        epochs (int): Number of training epochs.
        device (torch.device): Device (CPU/GPU).

    Returns:
        tuple: Minimum training loss, maximum training accuracy.
    """
    # Unfreeze the layers
    # freezing first layers 
    for mod in local_modalities:
        for param in model.sub_networks[mod].parameters():
            param.requires_grad = True  # Freezes the feature extractor
    model.to(device)
    model.train()

    training_losses = []
    training_accuracies = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch in train_loader:
            inputs, labels = batch

            # Prepare input data for selected modalities
            modality_inputs = {mod: inputs[mod].to(device) for mod in local_modalities}
            labels = labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(**modality_inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        # Compute loss and accuracy
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

        training_losses.append(avg_loss)
        training_accuracies.append(accuracy)

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    return min(training_losses), max(training_accuracies)