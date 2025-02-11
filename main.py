import argparse
import torch
from sheaffmtl import run_sheaf_fmtl
from dsgd import run_DSGD
from local_training import run_local_training
from utils import read_drone_6G, cross_entropy_loss_with_l2, generate_random_adj_mixing_matrices
from models import make_drone_classifier
import networkx as nx
import numpy as np
import random
import json
import torch.optim as optim




# Setting the seeds for reproducibilty
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def main():
    parser = argparse.ArgumentParser(description='Federated Learning Algorithms')
    parser.add_argument('--dataset', type=str, required=True, help='Input data (drone)')
    parser.add_argument('--algorithm', type=str, required=True, help='Algorithm name (sheaf-FMTL, DSGD, Local-training)')
    parser.add_argument('--alpha-g', type=float, default=0.001, help='Learning rate to update the task specific head of Sheaf-FMTL')
    parser.add_argument('--alpha-phi', type=float, default=0.001, help='Learning rate to update the modality specific feature extractor of Sheaf-FMTL')
    parser.add_argument('--eta', type=float, default=0.001, help='Learning rate to update the linear maps in Sheaf-FMTL')
    parser.add_argument('--local-iterations', type=int, default=1, help='Number of local iterations for dFedU')
    parser.add_argument('--local-lr', type=float, default=0.001, help='Local learning rate for dFedU')
    parser.add_argument('--lambda-reg', type=float, default=0.001, help='Regularization strength')
    parser.add_argument('--factor', type=float, default=0.2, help='Factor for P_ij matrix size for Sheaf-FMTL')
    parser.add_argument('--num-rounds', type=int, default=100, help='Number of training rounds')
    parser.add_argument('--times', type=int, default=1, help='Number of MC runs')
    parser.add_argument('--distribution', type=str, default='IID', help='Dataset distribution (IID, NIID)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    args = parser.parse_args()

    if args.dataset == 'drone':
        no_users = 20
        train_loaders, val_loaders, test_loaders, user_modalities, available_modalities, user_groups, group_definitions = read_drone_6G(distribution=args.distribution, batchsize=args.batch_size, no_users=no_users)
        all_models = []
        classifier_optimizers = []
        optimizers = []

        for i in range(no_users):
            model = make_drone_classifier(user_modalities[i])
            local_optimizer = optim.Adam(model.parameters(), lr=args.alpha_phi)
            class_optim = optim.Adam(model.classifier.parameters(), lr=args.alpha_g)
            all_models.append(model)
            optimizers.append(local_optimizer)
            classifier_optimizers.append(class_optim)
        loss_func = cross_entropy_loss_with_l2
        metric_name = 'Accuracy'
        metric_func = lambda targets, predictions: torch.mean((targets == predictions).float())
    
    else:
        raise ValueError("Invalid dataset type.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    adjacency_matrix, mixing_matrices = generate_random_adj_mixing_matrices(available_modalities, user_modalities, no_users)

    for time in range(args.times):
        
        if args.algorithm == 'Sheaf-FMTL':
            results_dict = run_sheaf_fmtl(train_loaders,
                                        val_loaders,
                                        optimizers,
                                        classifier_optimizers,
                                        user_groups,
                                        group_definitions,
                                        available_modalities,
                                        user_modalities,
                                        adjacency_matrix,
                                        mixing_matrices,
                                        args.num_rounds,
                                        args.local_iterations,
                                        args.alpha_g,
                                        args.alpha_phi,
                                        args.eta, 
                                        args.lambda_reg, 
                                        args.factor, 
                                        all_models, 
                                        loss_func,
                                        device
                                        )
        elif args.algorithm == "DSGD":
            results_dict = run_DSGD(train_loaders,
                                        val_loaders,
                                        optimizers,
                                        user_groups,
                                        group_definitions,
                                        available_modalities,
                                        user_modalities,
                                        adjacency_matrix,
                                        mixing_matrices,
                                        args.num_rounds,
                                        args.local_iterations,
                                        args.alpha_phi,
                                        all_models, 
                                        loss_func,
                                        device
                                        )

        else:
            raise ValueError('Invalid algorithm. Choose either "dFedU", "Sheaf-FL", Sheaf-FMTL-subgraph, or Local-training.')

        #print(f'Average Test {metric_name}s:', average_test_metrics)
        #print('Transmitted Bits per Iteration:', transmitted_bits_per_iteration)
        # Convert to numpy arrays
        #average_test_metrics_array = np.array(average_test_metrics)
        #transmitted_bits_per_iteration_array = np.array(transmitted_bits_per_iteration)
        

        # Construct the filename with parameters
        #alg = f"{args.dataset}_{args.algorithm}_{args.local_lr}_{args.alpha_g}_{args.eta}_{args.local_iterations}_{args.lambda_reg}b_{args.local_iterations}_{args.times}"
        alg = (
            f"{args.dataset}_{args.algorithm}_{args.local_lr}_{args.alpha_g}_{args.alpha_phi}_"
            f"{args.eta}_{args.local_iterations}_{args.lambda_reg}_factor_{args.factor}_"
            f"{args.num_rounds}_time_{time}"
        )


        with open(f"{alg}.json", "w") as f:
            json.dump(results_dict, f)  

if __name__ == '__main__':
    main()