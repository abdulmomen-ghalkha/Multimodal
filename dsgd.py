import torch
from utils import decentralized_aggregation, train_local_model

def run_DSGD(train_loaders, val_loaders,optimizers,user_groups,group_definitions,available_modalities,user_modalities,adjacency_matrix,mixing_matrices,num_rounds,local_iterations,alpha_phi,all_models, loss_func,device):
    # Dictionaries to store metrics
    group_train_loss_histories = {1: [], 2: [], 3: []}
    group_train_accuracy_histories = {1: [], 2: [], 3: []}
    group_val_loss_histories = {1: [], 2: [], 3: []}
    group_val_accuracy_histories = {1: [], 2: [], 3: []}

    global_rounds = num_rounds
    local_epochs = local_iterations
    no_users = len(train_loaders)
    # Decentralized Federated Learning Loop
    for round_num in range(global_rounds):

        # Decentralized aggregation
        decentralized_aggregation(all_models, mixing_matrices, available_modalities)
        epoch_group_train_losses = {1: [], 2: [], 3: []}
        epoch_group_train_accuracies = {1: [], 2: [], 3: []}
        epoch_group_val_losses = {1: [], 2: [], 3: []}
        epoch_group_val_accuracies = {1: [], 2: [], 3: []}

        
        print(f"Global Round {round_num + 1}")

        # Training for image modalities
        
        epoch_train_loss = 0.0
        correct_train_predictions = 0
        total_train_samples = 0

        for user_id in range(no_users):
            #print(f"Training model for User {user_id + 1}")
            print(user_id)
            user_models = all_models[user_id]
            group = user_groups[user_id]
            
            # Initialize optimizers for each modality
            
            # Train local model
            train_losses, train_accuracies = train_local_model(
                user_modalities[user_id],
                user_models,
                train_loaders[user_id],
                loss_func,
                optimizers[user_id],
                local_epochs
            )
            epoch_group_train_losses[group].append(train_losses)
            epoch_group_train_accuracies[group].append(train_accuracies)

            # Assign metrics to the respective group
            


        
        # Validation phase
        
        for user_id in range(no_users):
            user_models = all_models[user_id]
            val_dict = validate_user_models(
                user_id,
                user_models,
                val_loaders[user_id],
                criterion
            )
            group = user_groups[user_id]
            epoch_group_val_losses[group].append(val_dict["loss"])
            epoch_group_val_accuracies[group].append(val_dict["accuracy"])

        # Print group-wise metrics
        #print(f"Metrics for Global Round {round_num + 1}:")
        for group in [1, 2, 3]:

            group_train_loss_histories[group].append(epoch_group_train_losses[group])
            group_train_accuracy_histories[group].append(epoch_group_train_accuracies[group])
            group_val_loss_histories[group].append(epoch_group_val_losses[group])
            group_val_accuracy_histories[group].append(epoch_group_val_accuracies[group])

        print(f"{global_rounds}] Group Metrics:")
        for group in [1, 2, 3]:
            print(f"  Group {group} - Train Loss: {np.mean(group_train_loss_histories[group][-1]):.4f}, Train Accuracy: {np.mean(group_train_accuracy_histories[group][-1]):.4f}")
            print(f"  Group {group} - Val Loss: {np.mean(group_val_loss_histories[group][-1]):.4f}, Val Accuracy: {np.mean(group_val_accuracy_histories[group][-1]):.4f}")



    return 