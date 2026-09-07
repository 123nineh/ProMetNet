import itertools
import re
from typing import Union
import networkx as nx
import numpy as np
import pandas as pd
from .pathway_integrator import PathwayDataIntegrator


class PathwayGraph:
    """
    Build a directed graph network integrating proteomic and metabolomic features with pathways.

    This class aligns multi-omics data, maps features to pathways, optionally filters
    for relevant pathways, and constructs a directed network for downstream analysis.

    Args:
        input_prot_data (pd.DataFrame): Proteomics data.
        input_meta_data (pd.DataFrame): Metabolomics data.
        prot_group (pd.DataFrame): Proteomics sample groups.
        meta_group (pd.DataFrame): Metabolomics sample groups.
        pathways (pd.DataFrame): Source-target pathway interactions.
        mapping (pd.DataFrame or None, optional): Feature-to-pathway mapping. Default is None.
        input_data_column (str, optional): Column with feature names. Default is 'Features'.
        subset_pathways (bool, optional): Keep only pathways relevant to input features. Default is True.
        source_column (str, optional): Source column name in `pathways`. Default is 'source'.
        target_column (str, optional): Target column name in `pathways`. Default is 'target'.

    Attributes:
        input_data (pd.DataFrame): Aligned input data.
        mapping (pd.DataFrame): Feature-to-pathway mapping.
        pathways (pd.DataFrame): Filtered pathway data.
        inputs (list): Unique features in the network.
        netx (networkx.DiGraph): Directed pathway network.
    """
    def __init__(
        self,
        input_prot_data: pd.DataFrame,
        input_meta_data: pd.DataFrame,
        prot_group: pd.DataFrame,
        meta_group: pd.DataFrame,
        pathways: pd.DataFrame,
        mapping: Union[pd.DataFrame, None] = None,
        input_data_column: str = "Features",
        subset_pathways: bool = True,
        source_column: str = "source",
        target_column: str = "target",
    ):
        self.prot_data = input_prot_data
        self.meta_data = input_meta_data
        self.prot_group = prot_group
        self.meta_group = meta_group
        self.transaltion = mapping

        input_data, group_data = self.Integrator()
        self.input_data_column = input_data_column
        pathways = pathways.rename(
            columns={source_column: "source", target_column: "target"}
        )

        if isinstance(mapping, pd.DataFrame):
            self.mapping = mapping
            self.unaltered_mapping = mapping

        else:
            self.mapping = pd.DataFrame(
                {
                    "input": input_data[input_data_column].values,
                    "translation": input_data[input_data_column].values,
                }
            )
            self.unaltered_mapping = mapping

        if subset_pathways:
            self.mapping = _subset_input(input_data, self.mapping, input_data_column)

            self.pathways = _subset_pathways_on_idx(pathways, self.mapping)

        else:
            self.pathways = pathways

        self.mapping = _get_mapping_to_all_layers(self.pathways, self.mapping)

        self.input_data = input_data

        self.inputs = self.mapping["input"].unique()

        self.netx = self.build_network()


    def Integrator(self):

        integrator = PathwayDataIntegrator(
            prot_data=self.prot_data,
            meta_data=self.meta_data,
            prot_group_data=self.prot_group,
            meta_group_data=self.meta_group,
            translation_data=self.transaltion,
            output_dir = None
        )

        integrator.run_all()
        expression_matrix, group_matrix = integrator.get_results()
        return expression_matrix, group_matrix

    def build_network(self):
        """
        Build a directed graph from the object's pathway edges, adding a root node to connect all root nodes.

        This method generates a NetworkX DiGraph representing the pathway network,
        enabling downstream analyses such as subgraph extraction and network metrics.

        Returns:
            networkx.DiGraph: Directed graph of the pathway network.
        """
        if hasattr(self, "netx"):
            return self.netx

        net = nx.from_pandas_edgelist(
            self.pathways, source="target", target="source", create_using=nx.DiGraph()
        )
        roots = [n for n, d in net.in_degree() if d == 0]
        root_node = "root"
        edges = [(root_node, n) for n in roots]
        net.add_edges_from(edges)

        return net

    def get_layers(self, n_levels, direction="root_to_leaf") -> list:
        """
        Return pathway information organized by network levels.

        This method extracts pathways at each level of the network, along with their input features,
        allowing hierarchical analysis of the network structure.

        Args:
            n_levels (int): Number of levels below the root node to include.
            direction (str, optional): Layer direction, either "root_to_leaf" or "leaf_to_root". Default is "root_to_leaf".

        Returns:
            list of dict: Each dict maps pathway names to their corresponding input features for that level.
        """
        if direction == "root_to_leaf":
            net = _complete_network(self.netx, n_levels=n_levels)
            layers = _get_layers_from_net(net, n_levels)

        terminal_nodes = [n for n, d in net.out_degree() if d == 0]

        mapping_df = self.mapping
        dict = {}
        missing_pathways = []
        for p in terminal_nodes:
            pathway_name = re.sub("_copy.*", "", p)
            inputs = (
                mapping_df[mapping_df["connections"] == pathway_name]["input"]
                .unique()
                .tolist()
            )
            if len(inputs) == 0:
                missing_pathways.append(pathway_name)
            dict[pathway_name] = inputs
        layers.append(dict)
        return layers

    def get_connectivity_matrices(self, n_levels, direction="root_to_leaf") -> list:
        """
        Return connectivity matrices for each layer of the network.

        This method generates the connectivity matrices of all layers in the network,
        ordered according to the specified direction, to support hierarchical network analysis.

        Args:
            n_levels (int): Number of levels below the root node to include.
            direction (str, optional): Layer order, either "root_to_leaf" or "leaf_to_root". Default is "root_to_leaf".

        Returns:
            list of pd.DataFrame: Connectivity matrices for each network layer.
        """
        connectivity_matrices = []
        layers = self.get_layers(n_levels, direction)
        for i, layer in enumerate(layers[::-1]):
            layer_map = _get_map_from_layer(layer)
            if i == 0:
                inputs = list(layer_map.index)
                self.inputs = sorted(inputs)
            filter_df = pd.DataFrame(index=inputs)
            all = filter_df.merge(
                layer_map, right_index=True, left_index=True, how="inner"
            )
            all = all.reindex(sorted(all.columns), axis=1)
            all = all.sort_index()
            inputs = list(layer_map.columns)
            connectivity_matrices.append(all)
        return connectivity_matrices


def _get_mapping_to_all_layers(pathways, mapping):
    graph = nx.from_pandas_edgelist(
        pathways, source="source", target="target", create_using=nx.DiGraph()
    )
    components = {"input": [], "connections": []}
    for translation in mapping["input"]:
        ids = mapping[mapping["input"] == translation]["translation"]
        for id in ids:
            if graph.has_node(id):
                connections = graph.subgraph(
                    nx.single_source_shortest_path(graph, id).keys()
                ).nodes
                for connection in connections:
                    components["input"].append(translation)
                    components["connections"].append(connection)
    components = pd.DataFrame(components)
    components.drop_duplicates(inplace=True)
    return components


def _get_map_from_layer(layer_dict):
    pathways = layer_dict.keys()
    inputs = list(itertools.chain.from_iterable(layer_dict.values()))
    inputs = list(np.unique(inputs))
    df = pd.DataFrame(index=pathways, columns=inputs)
    for k, v in layer_dict.items():
        df.loc[k, v] = 1
    df = df.fillna(0)
    return df.T


def _add_edges(G, node, n_levels):
    edges = []
    source = node
    for level in range(n_levels):
        target = node + "_copy" + str(level + 1)
        edge = (source, target)
        source = target
        edges.append(edge)

    G.add_edges_from(edges)
    return G


def _complete_network(G, n_levels=4):
    nr_copies = 0
    sub_graph = nx.ego_graph(G, "root", radius=n_levels)
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]
    for node in terminal_nodes:
        distance = len(nx.shortest_path(sub_graph, source="root", target=node))
        if distance <= n_levels:
            nr_copies = nr_copies + n_levels - distance
            diff = n_levels - distance + 1
            sub_graph = _add_edges(sub_graph, node, diff)
    return sub_graph


def _get_nodes_at_level(net, distance):
    nodes = set(nx.ego_graph(net, "root", radius=distance))
    if distance >= 1.0:
        nodes -= set(nx.ego_graph(net, "root", radius=distance - 1))
    return list(nodes)


def _get_layers_from_net(net, n_levels):
    layers = []
    for i in range(n_levels):
        nodes = _get_nodes_at_level(net, i)
        dict = {}
        for n in nodes:
            n_name = re.sub("_copy.*", "", n)
            next = net.successors(n)
            dict[n_name] = [re.sub("_copy.*", "", nex) for nex in next]
        layers.append(dict)
    return layers


def _subset_input(
    input_df,
    translation,
    input_data_column,
):
    keys_in_data = input_df[input_data_column].unique()
    translation = translation[translation["input"].isin(keys_in_data)]
    return translation


def _subset_pathways_on_idx(pathways, translation):
    def add_pathways(idx_list, target):
        if len(target) == 0:
            return idx_list
        else:
            idx_list = idx_list + target
            subsetted_pathway = pathways[pathways["source"].isin(target)]
            new_target = list(subsetted_pathway["target"].unique())
            return add_pathways(idx_list, new_target)

    original_target = list(translation["translation"].unique())
    idx_list = []
    idx_list = add_pathways(idx_list, original_target)
    pathways = pathways[pathways["source"].isin(idx_list)]
    return pathways

