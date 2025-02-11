import copy
import torch
import torch.nn as nn
import numpy as np
from utils import validate_user_models, fmtl_aggregation, per_modelaity_decentralized_aggregation, train_local_model


def run_sheaf_fmtl(train_loaders, val_loaders, optimizers, classifier_optimizers, user_groups, group_definitions, available_modalities, user_modalities, adjacency_matrix, mixing_matrices, num_rounds, local_iterations, alpha_g, alpha_phi, eta, lambda_reg, factor, all_models, loss_func, device):

    # Dictionaries to store metrics
    group_train_loss_histories = {1: [], 2: [], 3: []}
    group_train_accuracy_histories = {1: [], 2: [], 3: []}
    group_val_loss_histories = {1: [], 2: [], 3: []}
    group_val_accuracy_histories = {1: [], 2: [], 3: []}
    

    global_rounds = num_rounds
    local_epochs = local_iterations
    no_users = len(train_loaders)
    neighbors = [np.nonzero(adjacency_matrix[i])[0].tolist() for i in range(no_users)]

    # Initialize P_ij matrices
    # Initialize P_ij matrices
    P = {}
    for i, j in zip(*adjacency_matrix.nonzero()):
        num_params_i = sum(p.numel() for p in all_models[i].classifier.parameters())
        num_params_j = sum(p.numel() for p in all_models[j].classifier.parameters())
        P[(i, j)] = torch.eye(int(factor*(num_params_i + num_params_j) // 2), num_params_i).to(device)
        P[(j, i)] = torch.eye(int(factor*(num_params_i + num_params_j) // 2), num_params_j).to(device)


    # Decentralized Federated Learning Loop
    for round_num in range(global_rounds):
        print(f"Global Round {round_num + 1}")

        # Decentralized aggregation step

        # Temporary storage for this round
        epoch_group_train_losses = {1: [], 2: [], 3: []}
        epoch_group_train_accuracies = {1: [], 2: [], 3: []}
        epoch_group_val_losses = {1: [], 2: [], 3: []}
        epoch_group_val_accuracies = {1: [], 2: [], 3: []}

        print(user_groups)
        per_modelaity_decentralized_aggregation(all_models, mixing_matrices, available_modalities, user_modalities)

        fmtl_aggregation(all_models, classifier_optimizers, train_loaders, user_modalities, user_groups, no_users, adjacency_matrix, P, alpha_g, lambda_reg, eta, loss_func, device)

        print("FinishedFMTLAggregatin")
        # Training phase
        for user_id in range(no_users):
            print(f"Training model for User {user_id + 1}")
            user_models = all_models[user_id]
            group = user_groups[user_id]

            # Train local model for the user's available modalities

            train_loss, train_accuracy = train_local_model(
                user_modalities[user_id], 
                user_models, 
                train_loaders[user_id], 
                loss_func, 
                optimizers[user_id], 
                local_epochs, 
                device
            )

    

            # Store in group-wise metrics
            epoch_group_train_losses[group].append(train_loss)
            epoch_group_train_accuracies[group].append(train_accuracy)


        

        # Validation phase
        for user_id in range(no_users):
            user_models = all_models[user_id]
            val_dict = validate_user_models(
                user_id, 
                user_models, 
                val_loaders[user_id], 
                loss_func, 
                user_modalities[user_id], 
                device
            )
            group = user_groups[user_id]
            epoch_group_val_losses[group].append(val_dict["loss"])
            epoch_group_val_accuracies[group].append(val_dict["accuracy"])

        # Store final metrics for each group
        for group in [1, 2, 3]:
            group_train_loss_histories[group].append(np.mean(epoch_group_train_losses[group]))
            group_train_accuracy_histories[group].append(np.mean(epoch_group_train_accuracies[group]))
            group_val_loss_histories[group].append(np.mean(epoch_group_val_losses[group]))
            group_val_accuracy_histories[group].append(np.mean(epoch_group_val_accuracies[group]))

        # Print final results for this round
        print(f"---- Global Round {round_num + 1} Metrics ----")
        for group in [1, 2, 3]:
            print(f"  Group {group} - Train Loss: {group_train_loss_histories[group][-1]:.4f}, Train Accuracy: {group_train_accuracy_histories[group][-1]:.4f}")
            print(f"  Group {group} - Val Loss: {group_val_loss_histories[group][-1]:.4f}, Val Accuracy: {group_val_accuracy_histories[group][-1]:.4f}")


    num_epochs = global_rounds
    # Convert metrics to numpy arrays for easy manipulation
    group_train_loss_histories = {k: np.array(v) for k, v in group_train_loss_histories.items()}
    group_train_accuracy_histories = {k: np.array(v) for k, v in group_train_accuracy_histories.items()}
    group_val_loss_histories = {k: np.array(v) for k, v in group_val_loss_histories.items()}
    group_val_accuracy_histories = {k: np.array(v) for k, v in group_val_accuracy_histories.items()}

    # Handle potential one-dimensional arrays
    group_train_loss_mean = {k: v.mean(axis=1) if v.ndim > 1 else v for k, v in group_train_loss_histories.items()}
    group_train_loss_std = {k: v.std(axis=1) if v.ndim > 1 else np.zeros_like(v) for k, v in group_train_loss_histories.items()}
    group_val_loss_mean = {k: v.mean(axis=1) if v.ndim > 1 else v for k, v in group_val_loss_histories.items()}
    group_val_loss_std = {k: v.std(axis=1) if v.ndim > 1 else np.zeros_like(v) for k, v in group_val_loss_histories.items()}

    group_train_acc_mean = {k: v.mean(axis=1) if v.ndim > 1 else v for k, v in group_train_accuracy_histories.items()}
    group_train_acc_std = {k: v.std(axis=1) if v.ndim > 1 else np.zeros_like(v) for k, v in group_train_accuracy_histories.items()}
    group_val_acc_mean = {k: v.mean(axis=1) if v.ndim > 1 else v for k, v in group_val_accuracy_histories.items()}
    group_val_acc_std = {k: v.std(axis=1) if v.ndim > 1 else np.zeros_like(v) for k, v in group_val_accuracy_histories.items()}

    # Convert numpy arrays to lists for serialization
    data_to_save = {
        "group_train_loss_mean": {k: v.tolist() for k, v in group_train_loss_mean.items()},
        "group_train_loss_std": {k: v.tolist() for k, v in group_train_loss_std.items()},
        "group_val_loss_mean": {k: v.tolist() for k, v in group_val_loss_mean.items()},
        "group_val_loss_std": {k: v.tolist() for k, v in group_val_loss_std.items()},
        "group_train_acc_mean": {k: v.tolist() for k, v in group_train_acc_mean.items()},
        "group_train_acc_std": {k: v.tolist() for k, v in group_train_acc_std.items()},
        "group_val_acc_mean": {k: v.tolist() for k, v in group_val_acc_mean.items()},
        "group_val_acc_std": {k: v.tolist() for k, v in group_val_acc_std.items()}
    }

    return data_to_save
