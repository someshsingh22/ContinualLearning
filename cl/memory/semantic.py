import torch
import torch.nn as nn


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

    def forward(self, subgraph):
        """
        Classifier
        """
        raise NotImplementedError
