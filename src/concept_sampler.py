import os
import pickle
from typing import *
from collections import defaultdict
from enum import Enum

import numpy as np

import networkx as nx
from networkx import Graph
from networkx.classes.reportviews import NodeView
import json

class SamplingMethod(Enum):
    RANDOM = 1
    EDGE_CONSTRAINED_RANDOM = 2
    EDGE_CONSTRAINED_RANDOM_WALK = 3

class GraphConceptSampler:
    def __init__(self, mode: SamplingMethod, class_list: List[str], G: Graph = None, graph_path: str = None) -> None:
        """
        Initialize the GraphConceptSampler object.

        Args:
            mode (SamplingMethod): The sampling method to be used.
            class_list (List[str]): The list of class names.
            G (Graph, optional): The graph object. Defaults to None.
            graph_path (str, optional): The path to the graph file. Defaults to None.

        Raises:
            ValueError: If an invalid sampling method is provided or if neither G nor graph_path is provided.
        """
        match mode:
            case SamplingMethod.RANDOM:
                self.class_sampler = self._random_sample
            case SamplingMethod.EDGE_CONSTRAINED_RANDOM:
                self.class_sampler = self._edge_constrained_random_sample
            case SamplingMethod.EDGE_CONSTRAINED_RANDOM_WALK:
                self.class_sampler = self._edge_constrained_random_walk
            case _:
                raise ValueError("Invalid sampling method.")

        if G is None:
            if graph_path is None:
                raise ValueError("Either G or graph_path should be provided.")
            G = nx.read_gexf(graph_path)
        self.G = G
        self.class_list = class_list
        self.class_id = list(range(len(class_list)))

        self.partitions = self._group_nodes() # dict of class_name: (class_node, [attr_nodes])

    def _group_nodes(self) -> Dict[str, Tuple[NodeView, List[NodeView]]]:
        partitions = {class_name: (node, []) for node in self.G.nodes if node in self.class_list for class_name in self.class_list}
        for node in self.G.nodes:
            probs = node["p"]
            non_zero_probs = [i for i, prob in enumerate(probs) if prob > 0]
            for id in non_zero_probs:
                class_name = self.class_list[id]
                partitions[class_name][1].append(node)
        return partitions

    def sample(self, num_samples: Union[int, List[int]]) -> List[Tuple[NodeView, List[NodeView]]]:
        """
        Samples nodes from each class in the concept sampler.

        Args:
            num_samples (Union[int, List[int]]): The number of samples to be drawn from each class.
                If an integer is provided, the same number of samples will be drawn from each class.
                If a list of integers is provided, each integer represents the number of samples to be drawn
                from the corresponding class.

        Returns:
            List[Tuple[NodeView, List[NodeView]]]: A list of tuples, where each tuple contains the class node
                and a list of sampled nodes for that class.
        """
        if isinstance(num_samples, int):
            num_samples = [num_samples] * len(self.class_list)
        if len(num_samples) != len(self.class_list):
            raise ValueError("Number of samples should match the number of classes.")
        
        samples = []
        for class_name, num_sample in zip(self.class_list, num_samples):
            class_node, attr_nodes = self.partitions[class_name]
            sampled_nodes = self.class_sampler(class_node, attr_nodes, num_sample)
            samples.append((class_node, sampled_nodes))
        return samples

    def _random_sample(self, node: NodeView, node_list: List[NodeView], num_sample: int) -> List[NodeView]:
        probs = np.array([n["p"] for n in node_list])
        return np.random.choice(node_list, num_sample, p=probs, replace=False)
    
    def _edge_constrained_random_sample(self, node: NodeView, node_list: List[NodeView], num_sample: int) -> List[NodeView]:
        connected_nodes = [n for n in node_list if self.G.has_edge(node, n)]
        probs = np.array([n["p"] for n in connected_nodes])
        return np.random.choice(connected_nodes, num_sample, p=probs, replace=False)
    
    def _edge_constrained_random_walk(self, node: NodeView, node_list: List[NodeView], num_sample: int) -> List[NodeView]:
        sampled_nodes = []
        cur_node = node
        for _ in range(num_sample):
            neighbors = list(self.G.neighbors(cur_node))
            probs = np.array([n["p"] for n in neighbors])
            next_node = np.random.choice(neighbors, 1, p=probs)[0]
            sampled_nodes.append(next_node)
            cur_node = next_node
        return np.array(set(sampled_nodes))


class ConceptSampler:
    def __init__(self, occurance_matrix: Dict[int, Dict[Tuple[str], Dict[str, int]]]) -> None:
        """
        Initialize the ConceptSampler object. The occurance matrix should follow the format:
        {
            k: {
                combo_tuple: {
                    class_name: n
                }
            }
        }

        Args:
            occurance_matrix (Dict[int, Dict[str, Dict[Tuple[str], int]]]): The occurance matrix.
        """
        self.occur_mat = occurance_matrix

    def absolute_sample(self) -> List[Tuple[str, str, int]]:
        """
        Samples the concepts that are underrepresented in the dataset. The output is a complement
        of the occurance matrix as a list.

        Returns:
            List[Tuple[str, str, int]]: A list of tuples, where each tuple contains the class name,
                attribute combination, and the number of samples to be drawn.
        """
        complement_list = []
        k = max(self.occur_mat.keys())
        while k > 1:
            occur_mat = self.occur_mat[k]
            for concept_combo, class_dict in occur_mat.items():
                max_count = max(class_dict.values())
                for class_name, count in class_dict.items():
                    if count < max_count:
                        complement_list.append((class_name, concept_combo, max_count - count))
            # TODO: update the co-occurance count of smaller cliques after sampling k-cliques
            k -= 1
        return complement_list

    def probability_sample(self, num_samples: int) -> List[Tuple[str, str, int]]:
        """
        Samples the concepts based on the probability distribution of the occurance matrix. The output
        is a list of samples.

        Args:
            num_samples (int): The number of samples to be drawn.

        Returns:
            List[Tuple[str, str, int]]: A list of tuples, where each tuple contains the class name,
                attribute combination, and the number of samples to be drawn.
        """
        pass


class WaterbirdConceptSampler(ConceptSampler):
    def __init__(self, lb_path, wb_path) -> None:
        """
        Waterbird concept file is special. This initiator will load the waterbird and landbird occurances,
        sort by clique size, and store them in the desired dictionary form.
        """
        wb_occur = None
        lb_occur = None
        with open(wb_path, "rb") as f:
            wb_occur = pickle.load(f)
        with open(lb_path, "rb") as f:
            lb_occur = pickle.load(f)
        wb_occur_count = defaultdict(set)
        lb_occur_count = defaultdict(set)
        for k, v in wb_occur.items():
            wb_occur_count[len(k)].add((k, v))
        for k, v in lb_occur.items():
            lb_occur_count[len(k)].add((k, v))
        max_clique = max(max(wb_occur_count.keys()), max(lb_occur_count.keys()))
        self.occurs = OrderedDict()
        for i in range(1, max_clique + 1):
            self.occurs[i] = defaultdict(dict)
            wb_cliques = wb_occur_count[i]
            lb_cliques = lb_occur_count[i]
            for wb_clique, wb_count in wb_cliques:
                self.occurs[i][wb_clique]["waterbird"] = wb_count
            for lb_clique, lb_count in lb_cliques:
                self.occurs[i][lb_clique]["landbird"] = lb_count
        super().__init__(self.occurs)


class GenericConceptSampler(ConceptSampler):
    def __init__(self, occurance_file: str) -> None:
        """
        Initialize the GenericConceptSampler object.

        Args:
            occurance_file (str): The path to the occurance matrix file.
        """
        occurance_matrix = None
        with open(occurance_file, "rb") as f:
            occurance_matrix = pickle.load(f)
        super().__init__(occurance_matrix)
        

with open('/home/rwiddhi/datadebias/metadata/clique_dict_final_coco.pkl', 'rb') as f:
    clique_dict = pickle.load(f)


lst = ConceptSampler(clique_dict).absolute_sample()

with open('concepts_generation_coco.json', 'w') as file:
    json.dump(lst, file)
