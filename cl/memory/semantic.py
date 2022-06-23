from gettext import npgettext
import torch
import torch.nn as nn
import numpy as np
from utils.data_utils import load_pretrained_node2vec
from utils.link_predict import find_optimal_cutoff, link_prediction_eval
from utils.evaluation import run_evaluation_main
from finetuning import (
    run_finetuning_wkfl2,
    run_finetuning_wkfl3,
    setup_finetuning_input,
)

class GNN(nn.Module):
    def __init__(self, tokenizer, embedding, graph, vocab, args):
        super(GNN, self).__init__()
        self.args = args
        self.graph = self.get_graph(vocab)

    def get_graph(self, vocab):
        """
        vocab ={"cat"}
        for word in vocab:
            graph[word]= []
            for edge in load_concept(word)['edges']:
                graph[word].append((edge['end'], edge['weight'], edge['rel']))

        """

    def get_subgraph(self, task_vocab):
        # load json
        # node features from self.embedding
        # get graph
        return {k: v for k, v in self.graph.items() if k in task_vocab}

    def get_subgraph_embedding(self, subgraph) -> torch.Tensor:
        """
        subgraph = {'cat': [("mammal", 1, 'is_a')]}

        """
        raise NotImplementedError
        

    def forward(self, subgraph, node1, node2):
        """
        Classifier
        """
        # training_graph = subgraph
        # edge_tr = subgraph['edge']
        # vertex = subgraph['node']
        # node_embedding = GNN.get_subgraph_embedding(subgraph)

        #Link prediction
        ft_num_batches = setup_finetuning_input(args, attr_graph, context_gen)
        pred_data, true_data = run_finetuning_wkfl2(
        args, attr_graph, ft_num_batches, GNN.get_subgraph_embedding(), ent2id, rel2id)
        print("\n Begin evaluation for link prediction...")
        valid_true_data = np.array(true_data["valid"])
        threshold = find_optimal_cutoff(valid_true_data, pred_data["valid"])[0]
        run_evaluation_main(
        test_edges, pred_data["test"], true_data["test"], threshold, header="workflow2"
    )
    


        raise NotImplementedError
#sort 