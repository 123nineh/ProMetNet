from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split
from .prometnet import ProMetNet
import numpy as np
import torch
import pandas as pd
import lightning.pytorch as pl
import networkx as nx


class RecursivePathwayElimination:
    """
    RecursivePathwayElimination performs pathway-level feature selection using a ProMetNet model.

    This class implements a recursive pathway elimination strategy, where model predictions
    are interpreted with a specified explainer to identify and remove less important pathways
    iteratively. It provides a framework to understand which biological pathways contribute
    most to the model's predictions.

    Args:
        model (ProMetNet): Trained ProMetNet model used for pathway elimination.
        explainer: Explainer object for interpreting model predictions.

    Attributes:
        model (ProMetNet): The model used for pathway elimination.
        n_layers (int): Total number of layers in the model.
        learning_rate: Learning rate used during model updates.
        weight: Weight parameter(s) used in the model.
        explainer: Explainer for generating pathway importance scores.
        connectivity_matrices: Connectivity matrices defining pathway relationships in the model.
    """
    def __init__(self, model: ProMetNet, explainer):
        self.model = model
        self.n_layers = model.n_layers
        self.learning_rate = model.learning_rate
        self.weight = model.weight
        self.n_layers = model.n_layers
        self.explainer = explainer
        self.connectivity_matrices = model.connectivity_matrices

    def fit(
        self,
        input_data,
        design_matrix,
        nr_iterations: int = 20,
        max_epochs: int = 50,
        clip_threshold=1e-5,
        constant_removal_rate=0.05,
        min_features_per_layer=3,
        early_stopping=True,
    ):
        """
        Train the model and perform recursive pathway elimination.

        This method fits the ProMetNet model using the provided input data and design matrix,
        while iteratively performing recursive pathway elimination to identify the most important
        features. It supports early stopping and configurable parameters for robust and efficient
        feature selection across layers.

        Args:
            input_data: Data used for training the model.
            design_matrix: Design matrix corresponding to the input data.
            nr_iterations (int): Number of iterations for the elimination process.
            max_epochs (int): Maximum epochs for model training.
            clip_threshold: Threshold for clipping feature importance values.
            constant_removal_rate: Fraction of constant features removed per iteration.
            min_features_per_layer: Minimum number of features to retain per layer.
            early_stopping (bool): Whether to apply early stopping during training.

        Returns:
            dict: Results of the elimination process, including validation metrics, iteration info,
            trained models, trainable parameters, and connectivity matrices.
        """
        if early_stopping:
            print(f"Will apply early stopping")

        return_dict = {
            "models": [],
            "val_acc": [],
            "val_loss": [],
            "iteration": [],
            "epochs": [],
            "trainable_params": [],
            "matrices": [],
        }

        for iteration in range(nr_iterations):
            print(f"---------------- Iteration: {iteration} ----------------")

            self.fitted_input_data = self._fit_data_matrix_to_network_input(
                input_data.reset_index(),
                features=self.connectivity_matrices[0].index.values.tolist(),
            )

            splits = self._generate_k_folds(
                self.fitted_input_data, design_matrix=design_matrix
            )

            val_accs = []
            val_losses = []
            epochs = []
            for split in splits:
                self.model = ProMetNet(
                    connectivity_matrices=self.connectivity_matrices,
                    n_layers=self.n_layers,
                    dropout=0.2,
                    weight=self.weight,
                    validate=True,
                    residual=True,
                    scheduler="plateau",
                    learning_rate=self.learning_rate,
                )
                X_train, y_train, X_val, y_val = split
                train_dataloader = torch.utils.data.DataLoader(
                    dataset=torch.utils.data.TensorDataset(
                        torch.Tensor(X_train), torch.LongTensor(y_train)
                    ),
                    batch_size=8,
                    num_workers=12,
                    shuffle=True,
                )
                val_dataloader = torch.utils.data.DataLoader(
                    dataset=torch.utils.data.TensorDataset(
                        torch.Tensor(X_val), torch.LongTensor(y_val)
                    ),
                    batch_size=8,
                    num_workers=12,
                )
                callbacks = []
                if early_stopping:
                    callbacks.append(
                        early_stopping=(
                            pl.callbacks.early_stopping.EarlyStopping(
                                patience=10,
                                min_delta=0.001,
                                monitor="val_loss",
                                mode="min",
                            )
                        )
                    )

                trainer = pl.Trainer(
                    max_epochs=max_epochs,
                    enable_progress_bar=False,
                    enable_model_summary=False,
                    callbacks=callbacks,
                )
                trainer.fit(self.model, train_dataloader, val_dataloader)
                val_dict = trainer.validate(self.model, val_dataloader)
                val_accs.append(val_dict[0]["val_acc"])
                val_losses.append(val_dict[0]["val_loss"])
                epochs.append(self.model.current_epoch)

            return_dict["val_acc"].append(val_accs)
            return_dict["val_loss"].append(val_losses)
            return_dict["epochs"].append(epochs)
            return_dict["trainable_params"].append(self.model.trainable_params)
            return_dict["models"].append(self.model)
            return_dict["iteration"].append(iteration)

            self.test_data = torch.Tensor(np.concatenate([X_train, X_val], axis=0))
            self.background_data = torch.Tensor(
                np.concatenate([X_train, X_val], axis=0)
            )
            self.explainer.update_model(self.model)
            for wanted_layer in range(self.n_layers + 1):
                shap_dict = self.explainer._explain_layer(
                    self.test_data, self.background_data, wanted_layer
                )
                values = shap_dict["shap_values"]
                values = np.abs(np.array(values))
                values = np.mean(values, axis=1)
                values = np.sum(values, axis=0)
                shap_list = [
                    (feature, value)
                    for feature, value in zip(shap_dict["features"], values)
                ]
                if len(shap_list) <= min_features_per_layer:
                    print(f"Min features per layer ({min_features_per_layer}) reached")
                    return return_dict
                else:
                    shap_list = sorted(shap_list, key=lambda x: x[1])
                    features_below_clip = [
                        x[0] for x in shap_list if x[1] < clip_threshold
                    ]
                    n_features_below_clip = len(features_below_clip)
                    shap_list = shap_list[n_features_below_clip:]
                    total_n_elements = len(shap_list)
                    n_features_to_remove = int(total_n_elements * constant_removal_rate)
                    features_to_remove = [
                        x[0] for x in shap_list[:n_features_to_remove]
                    ]
                    features_to_remove += features_below_clip
                    features_to_remove = list(set(features_to_remove))
                    for feature in features_to_remove:
                        self.connectivity_matrices = _remove_node(
                            self.connectivity_matrices, feature
                        )

                    for matrix in self.connectivity_matrices:
                        print(matrix.shape)
                return_dict["matrices"].append(self.connectivity_matrices)

        return return_dict

    def get_final_model(self):
        return self.model

    def get_input_data(self):
        return self.fitted_input_data

    def _fit_data_matrix_to_network_input(
        self, data_matrix: pd.DataFrame, features, feature_column="Protein"
    ) -> pd.DataFrame:
        nr_features_in_matrix = len(data_matrix.index)
        if len(features) > nr_features_in_matrix:
            features_df = pd.DataFrame(features, columns=[feature_column])
            data_matrix = data_matrix.merge(features_df, how="right", on=feature_column)
        if len(features) > 0:
            data_matrix.set_index(feature_column, inplace=True)
            data_matrix = data_matrix.loc[features]
        return data_matrix

    def _generate_data(
        self,
        data_matrix: pd.DataFrame,
        design_matrix: pd.DataFrame,
        groups=[1, 2],
        test_size=0.25,
    ):
        y = []
        dfs = []
        for i, group in enumerate(groups):
            group_columns = design_matrix[design_matrix["group"] == group][
                "sample"
            ].values
            df = data_matrix[group_columns].T
            dfs.append(df)
            y += [i for _ in group_columns]
        y = np.array(y)
        X = pd.concat(dfs).fillna(0).to_numpy()
        X = preprocessing.StandardScaler().fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)
        return X_train, X_val, y_train, y_val

    def _generate_k_folds(
        self,
        data_matrix: pd.DataFrame,
        design_matrix: pd.DataFrame,
        groups=[1, 2],
        n_folds=3,
    ):
        y = []
        dfs = []
        for i, group in enumerate(groups):
            group_columns = design_matrix[design_matrix["group"] == group][
                "sample"
            ].values
            df = data_matrix[group_columns].T
            dfs.append(df)
            y += [i for _ in group_columns]
        y = np.array(y)
        X = pd.concat(dfs).fillna(0).to_numpy()
        X = preprocessing.StandardScaler().fit_transform(X)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
        splits = []
        for train_index, val_index in skf.split(X, y):
            X_train = X[train_index, :]
            y_train = y[train_index]
            X_val = X[val_index, :]
            y_val = y[val_index]
            splits.append((X_train, y_train, X_val, y_val))
        return splits


def _remove_node(connectivity_matrices, node_to_remove):
    G = nx.DiGraph()
    for matrix in connectivity_matrices:
        all_rows = matrix.index.values.tolist()
        all_columns = matrix.columns.values.tolist()
        rows, cols = np.where(matrix == 1)
        for r, c in zip(rows, cols):
            G.add_edge(all_rows[r], all_columns[c])
    """
    G is now a populated DiGraph
    """
    if not G.has_node(node_to_remove):
        return connectivity_matrices

    removed_successors = []
    for sources, targets in [
        n for n in nx.traversal.bfs_successors(G, node_to_remove, depth_limit=None)
    ]:
        for target in targets:
            if len(G.in_edges(target)) <= 1:
                G.remove_node(target)
                removed_successors.append(target)
        """
        we want to check if these nodes have more than one incoming node, else remove
        """

    G_reverse = G.reverse()
    removed_predecessors = []

    def remove_predecessor_nodes(node_to_remove):
        if not G.has_node(node_to_remove):
            return []
        removed_nodes = []
        for sources, targets in [
            n
            for n in nx.traversal.bfs_successors(
                G_reverse, node_to_remove, depth_limit=1
            )
        ]:
            for target in targets:
                if len(G.out_edges(target)) <= 1:
                    G.remove_node(target)
                    removed_nodes.append(target)
        return removed_nodes

    removed_nodes = remove_predecessor_nodes(node_to_remove)
    removed_predecessors += removed_nodes
    while len(removed_nodes) > 0:
        sum_removed_nodes = 0
        for node in removed_nodes:
            removed_nodes = remove_predecessor_nodes(node)
            sum_removed_nodes += len(removed_nodes)
            removed_predecessors += removed_nodes

    G.remove_node(node_to_remove)
    """
    Now we want to remove the sucecssors, predecessors, and node_to_remove from the connectivity matrices. 
    The successors should be removed from the connectivity matrix.
    """
    all_removed_nodes = removed_successors + removed_predecessors + [node_to_remove]
    new_matrices = []
    for matrix in connectivity_matrices:
        for node in all_removed_nodes:
            if node in matrix.index:
                matrix = matrix.drop(index=[node])
            if node in matrix.columns:
                matrix = matrix.drop(columns=[node])
        new_matrices.append(matrix)

    return new_matrices
