from typing import Union
from ProMetNet.plot import subgraph_sankey, complete_sankey
import networkx as nx
import numpy as np
import pandas as pd


class ImportanceNetwork:
    """
    Construct and analyze a directed graph representing the system's importance network.

    This class builds a directed graph from a dataframe of node-to-node importance flows,
    allowing downstream and upstream subgraph analysis, normalization, and visualization.
    It provides a framework to study how different nodes contribute to the overall system
    and supports multiple normalization strategies to quantify relative importance.

    Parameters:
        importance_df (pandas.DataFrame): DataFrame with columns for source node, target node, value flow, and layer.
            Represents the complete importance network.
        val_col (str, optional): Column name for the value flow between nodes. Defaults to "value".
        norm_method (str, optional): Method to normalize the value column. Options are:
            - 'SubgNorm': normalize by log(number of nodes in the subgraph for each node)
            - 'DegNorm': normalize by log(fan-in + fan-out) for each node.

    Attributes:
        importance_df (pandas.DataFrame): DataFrame used for constructing and plotting subgraphs.
        val_col (str): Column name representing value flow between nodes.
        G (networkx.DiGraph): Directed graph of the importance network.
        G_reverse (networkx.DiGraph): Directed graph of the network with edges reversed.
        norm_method (str): Normalization strategy applied to value flows.
    """
    def __init__(
        self,
        importance_df: pd.DataFrame,
        norm_method: str = "SubgNorm",
        val_col: str = "value",
    ):
        self.root_node = 0
        self.importance_df = importance_df
        self.val_col = val_col
        self.importance_graph = self.create_graph()
        self.importance_graph_reverse = self.importance_graph.reverse()
        self.norm_method = norm_method
        if norm_method:
            self.importance_df = self.add_normalization(method=norm_method)


    def plot_subgraph_sankey(
        self,
        query_node: str,
        upstream: bool = False,
        val_col: str = "value",
        cmap: str = "coolwarm"
    ):
        """
        Generate a Sankey diagram to visualize node flows in the importance network.

        This method creates a Sankey diagram starting from the specified query node,
        allowing visualization of either upstream or downstream flows. The width of
        links reflects the value flow between nodes, providing an intuitive view
        of how different nodes contribute to the network.

        Args:
            query_node (str): Node to serve as the starting point of the Sankey diagram.
            upstream (bool, optional): If True, shows upstream flows; if False (default), shows downstream flows.
            val_col (str, optional): Column in the DataFrame representing the value flow between nodes. Defaults to "value".
            cmap_name (str, optional): Color map name for the Sankey diagram. Defaults to "coolwarm".

        Returns:
            plotly.graph_objs._figure.Figure: Figure object of the Sankey diagram for visualization.
        """
        if upstream:
            final_node_id = self.get_node(query_node)
            subgraph = self.get_upstream_subgraph(final_node_id, depth_limit=None)
            source_or_target = "target"
        else:
            if query_node == "root":
                return ValueError("You cannot look downstream from root")
            final_node_id = self.get_node("root")
            query_node_id = self.get_node(query_node)
            subgraph = self.get_downstream_subgraph(self.importance_graph, query_node_id, depth_limit=None)
            source_or_target = "source"
            
        nodes_in_subgraph = [n for n in subgraph.nodes]

        df = self.importance_df[
            self.importance_df[source_or_target].isin(nodes_in_subgraph)
        ].copy()

        if df.empty:
            return ValueError("There are no nodes in the specified subgraph")

        fig = subgraph_sankey(
            df, final_node=final_node_id, val_col=val_col, cmap_name=cmap
        )
        return fig

    def plot_complete_sankey(
        self,
        multiclass: bool = False,
        show_top_n: int = 10,
        node_cmap: str = "Reds",
        edge_cmap: Union[str, list] = "Reds",
        save_file:str = 'None'
    ):
        """
        Plot a full Sankey diagram to visualize the importance network.

        This method generates a comprehensive Sankey diagram of the importance network,
        highlighting the relative contribution of each node and connection. It supports
        multiclass visualization and can display only the top N nodes to focus on
        key pathways. Node and edge colors can be customized to enhance interpretability.

        Parameters:
            multiclass (bool, optional): If True, plot a multiclass Sankey diagram. Defaults to False.
            show_top_n (int, optional): Display only the top N nodes in the diagram. Defaults to 10.
            node_cmap (str, optional): Color map for the nodes. Defaults to "Reds".
            edge_cmap (str or list, optional): Color map for the edges. Defaults to "Reds".

        Returns:
            plotly.graph_objs._figure.Figure: Figure object representing the Sankey diagram.
        """
        fig = complete_sankey(
            self.importance_df,
            multiclass=multiclass,
            val_col=self.val_col,
            show_top_n=show_top_n,
            edge_cmap=edge_cmap,
            node_cmap=node_cmap,
            save_file=save_file
        )
        return fig

    def create_graph(self):
        """
        Construct a directed graph (DiGraph) from node relationships in the input DataFrame.

        This method builds a directed graph using the source and target nodes provided
        in the dataframe. The resulting graph can be used for downstream network analysis,
        including path tracing, connectivity evaluation, and visualization of node importance.

        Returns:
            networkx.DiGraph: Directed graph representing the relationships between nodes.
        """
        importance_graph = nx.DiGraph()
        for k in self.importance_df.iterrows():
            source_name = k[1]["source name"]
            source = k[1]["source"]
            value = k[1][self.val_col]
            source_layer = k[1]["source layer"] + 1
            importance_graph.add_node(
                source, weight=value, layer=source_layer, name=source_name
            )
        for k in self.importance_df.iterrows():
            source = k[1]["source"]
            target = k[1]["target"]
            importance_graph.add_edge(source, target)
        root_layer = max(self.importance_df["target layer"]) + 1
        importance_graph.add_node(
            self.root_node, weight=0, layer=root_layer, name="root"
        )
        return importance_graph

    def get_downstream_subgraph(self, graph, query_node: str, depth_limit=None):
        """
        Extract a downstream subgraph from a specified query node.

        This method generates a directed subgraph containing all nodes downstream of
        the given query node, optionally limited by a maximum depth. The resulting subgraph
        can be used to analyze downstream influences, connectivity, and feature importance
        within the network.

        Args:
            query_node (str): Name of the node from which the downstream subgraph is constructed.
            depth_limit (int, optional): Maximum depth to include in the subgraph. Defaults to None.

        Returns:
            networkx.DiGraph: Directed subgraph containing downstream nodes up to the specified depth.
        """
        subgraph = nx.DiGraph()
        nodes = [
            n
            for n in nx.traversal.bfs_successors(
                graph, query_node, depth_limit=depth_limit
            )
            if n != query_node
        ]
        for source, targets in nodes:
            subgraph.add_node(source, **graph.nodes()[source])
            for t in targets:
                subgraph.add_node(t, **graph.nodes()[t])
        for node1 in subgraph.nodes():
            for node2 in subgraph.nodes():
                if graph.has_edge(node1, node2):
                    subgraph.add_edge(node1, node2)

        subgraph.add_node(query_node, **graph.nodes()[query_node])
        return subgraph

    def get_upstream_subgraph(self, query_node: str, depth_limit=None):
        """
        Extract an upstream subgraph from a specified query node.

        This method generates a directed subgraph containing all nodes upstream of
        the given query node, optionally limited by a maximum depth. The resulting subgraph
        can be used to analyze upstream influences, connectivity, and feature importance
        within the network.

        Args:
            query_node (str): Name of the node from which the upstream subgraph is constructed.
            depth_limit (int, optional): Maximum depth to include in the subgraph. Defaults to None.

        Returns:
            networkx.DiGraph: Directed subgraph containing upstream nodes up to the specified depth.
        """
        subgraph = self.get_downstream_subgraph(self.importance_graph_reverse, query_node, depth_limit=depth_limit)
        return subgraph

    def get_complete_subgraph(self, query_node: str, depth_limit=None):
        """
        Extract a complete subgraph containing both upstream and downstream nodes from a query node.

        This method generates a directed subgraph that includes all nodes influencing
        or influenced by the specified query node, optionally limited by a maximum depth.
        The resulting subgraph can be used to analyze full network context, connectivity,
        and feature importance surrounding the node of interest.

        Args:
            query_node (str): Name of the node from which the complete subgraph is constructed.
            depth_limit (int, optional): Maximum depth to include in the subgraph. Defaults to None.

        Returns:
            networkx.DiGraph: Directed subgraph containing both upstream and downstream nodes
            up to the specified depth.
        """
        subgraph = self.get_downstream_subgraph(self.importance_graph, query_node, depth_limit=depth_limit)
        nodes = [
            n
            for n in nx.traversal.bfs_successors(
                self.importance_graph_reverse, query_node, depth_limit=depth_limit
            )
            if n != query_node
        ]
        for source, targets in nodes:
            subgraph.add_node(source, **self.importance_graph_reverse.nodes()[source])
            for t in targets:
                subgraph.add_node(t, **self.importance_graph_reverse.nodes()[t])
        for node1 in subgraph.nodes():
            for node2 in subgraph.nodes():
                if self.importance_graph_reverse.has_edge(node1, node2):
                    subgraph.add_edge(node1, node2)

        subgraph.add_node(
            query_node, **self.importance_graph_reverse.nodes()[query_node]
        )
        return subgraph

    def get_nr_nodes_in_upstream_subgraph(self, query_node: str):
        """
        Compute the number of nodes in the upstream subgraph of a query node.

        This method calculates how many nodes influence the specified query node
        by examining all upstream connections. This metric helps quantify the
        extent of upstream influence within the network.

        Args:
            query_node (str): Name of the node from which the upstream subgraph is analyzed.

        Returns:
            int: Number of nodes in the upstream subgraph of the query node.
        """
        subgraph = self.get_upstream_subgraph(query_node, depth_limit=None)
        return subgraph.number_of_nodes()

    def get_nr_nodes_in_downstream_subgraph(self, query_node: str):
        """
        Compute the number of nodes in the downstream subgraph of a query node.

        This method calculates how many nodes are influenced by the specified query node
        by examining all downstream connections. This metric helps quantify the
        extent of downstream influence within the network.

        Args:
            query_node (str): Name of the node from which the downstream subgraph is analyzed.

        Returns:
            int: Number of nodes in the downstream subgraph of the query node.
        """
        subgraph = self.get_downstream_subgraph(self.importance_graph, query_node, depth_limit=None)
        return subgraph.number_of_nodes()

    def get_fan_in(self, query_node: str):
        """
        Compute the number of incoming edges (fan-in) for a query node.

        This method calculates how many edges point to the specified query node,
        providing a measure of the node's upstream connectivity and influence
        within the network.

        Args:
            query_node (str): Name of the node for which to calculate fan-in.

        Returns:
            int: Number of incoming edges (fan-in) for the query node.
        """
        return len([n for n in self.importance_graph.in_edges(query_node)])

    def get_fan_out(self, query_node: str):
        """
        Compute the number of outgoing edges (fan-out) for a query node.

        This method calculates how many edges originate from the specified query node,
        providing a measure of the node's downstream connectivity and influence
        within the network.

        Args:
            query_node (str): Name of the node for which to calculate fan-out.

        Returns:
            int: Number of outgoing edges (fan-out) for the query node.
        """
        return len([n for n in self.importance_graph.out_edges(query_node)])

    def add_normalization(self, method: str = "SubgNorm"):
        """
        Normalize importance values in the network based on the specified method.

        This method applies normalization to the importance values of nodes to allow
        comparisons across different scales or layers. The "DegNorm" method adjusts values
        based on each node's fan-in and fan-out, while the "SubgNorm" method considers
        the size of upstream and downstream subgraphs, providing a context-aware measure
        of relative importance.

        Args:
            method (str): Normalization method to apply. Options are:
                - "DegNorm": normalize based on fan-in and fan-out values.
                - "SubgNorm": normalize based on the number of nodes in upstream and downstream subgraphs.

        Returns:
            pd.DataFrame: DataFrame containing the original importance values along with the normalized values.
        """
        if method == "DegNorm":
            fan_in = np.array([self.get_fan_in(x) for x in self.importance_df["source"]])
            fan_out = np.array([self.get_fan_out(x) for x in self.importance_df["source"]])
            nr_tot = fan_in + fan_out + 1
        if method == "SubgNorm":
            upstream_nodes = np.array(
                [
                    self.get_nr_nodes_in_upstream_subgraph(x)
                    for x in self.importance_df["source"]
                ]
            )
            downstream_nodes = np.array(
                [
                    self.get_nr_nodes_in_downstream_subgraph(x)
                    for x in self.importance_df["source"]
                ]
            )
            nr_tot = upstream_nodes + downstream_nodes

        self.importance_df["value"] = self.importance_df["value"] / np.log2(nr_tot)
        return self.importance_df

    def get_node(self, name):
        for node, d in self.importance_graph.nodes(data=True):
            if d["name"] == name:
                return node
        raise ValueError(f"Could not find node {name}")
