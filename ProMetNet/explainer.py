import shap
import numpy as np
import torch
from .prometnet import ProMetNet
import pandas as pd
import lightning.pytorch as pl
from .feature_selection import RecursivePathwayElimination


class ProMetNetExplainer:
    """
    Provides feature-level explanations for ProMetNet predictions using SHAP.

    Args:
        model (ProMetNet): The trained model whose outputs will be interpreted.

    Notes:
        This class computes SHAP values for input samples and helps identify
        important proteins or metabolites driving the predictions.
    """
    def __init__(self, model: ProMetNet):
        self.model = model

    def update_model(self, model: ProMetNet):
        self.model = model

    def explain(self, test_data: torch.tensor, background_data: torch.tensor):
        """
        Compute and summarize SHAP feature importances for ProMetNet predictions.

        This method calculates Shapley values for each feature based on the provided
        background data, aggregates the importances, and returns a DataFrame
        that highlights the most influential proteins or metabolites.

        Args:
            test_data (torch.Tensor): Samples to generate explanations for.
            background_data (torch.Tensor): Reference data for SHAP calculation.

        Returns:
            pd.DataFrame: Aggregated feature importances, suitable for visualization or downstream analysis.
        """
        shap_dict = self._explain_layers(background_data, test_data)
        feature_dict = {
            "source": [],
            "target": [],
            "source name": [],
            "target name": [],
            "value": [],
            "type": [],
            "source layer": [],
            "target layer": [],
        }
        connectivity_matrices = self.model.get_connectivity_matrices()
        feature_id_mapping = {}

        feature_id = 0
        feature_id_mapping["root"] = feature_id
        for layer_features in shap_dict["features"]:
            for feature in layer_features:
                feature_id += 1
                feature_id_mapping[feature] = feature_id

        curr_layer = 0
        for sv, features, cm in zip(
            shap_dict["shap_values"], shap_dict["features"], connectivity_matrices
        ):
            sv = np.asarray(sv)
            sv = abs(sv)
            sv_mean = np.mean(sv, axis=1)

            for feature in range(sv_mean.shape[-1]):
                n_classes = sv_mean.shape[0]
                connections = cm[cm.index == features[feature]]
                connections = connections.loc[
                    :, (connections != 0).any(axis=0)
                ]
                for target in connections:
                    for curr_class in range(n_classes):
                        feature_dict["source"].append(
                            feature_id_mapping[features[feature]]
                        )
                        feature_dict["target"].append(feature_id_mapping[target])
                        feature_dict["source name"].append(features[feature])
                        feature_dict["target name"].append(target)
                        feature_dict["value"].append(sv_mean[curr_class][feature])
                        feature_dict["type"].append(curr_class)
                        feature_dict["source layer"].append(curr_layer)
                        feature_dict["target layer"].append(curr_layer + 1)
            curr_layer += 1
        df = pd.DataFrame(data=feature_dict)
        return df

    def fast_train(self, dataloader, num_epochs, optimizer):
        return_dict = {"accuracies":[], "losses":[], "epoch":[]}
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            total_accuracy = 0

            for _, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(self.model.device)
                targets = targets.to(self.model.device).type(torch.LongTensor)
                optimizer.zero_grad()
                outputs = self.model(inputs).to(self.model.device)
                loss = torch.nn.functional.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_accuracy += torch.sum(
                    torch.argmax(outputs, axis=1) == targets
                ) / len(targets)

            avg_loss = total_loss / len(dataloader)
            avg_accuracy = total_accuracy / len(dataloader)
            return_dict["accuracies"].append(avg_accuracy.numpy().tolist())
            return_dict["losses"].append(avg_loss)
            return_dict["epoch"].append(epoch)
        print(
            f"Final epoch: Average Accuracy {avg_accuracy:.2f}, Average Loss: {avg_loss:.2f}"
        )
        return self.model, return_dict

    def explain_average(
        self,
        test_data: torch.Tensor,
        background_data: torch.Tensor,
        nr_iterations: int,
        max_epochs: int,
        dataloader,
        fast_train: bool,
    ) -> (pd.DataFrame, dict):
        """
        Generate robust SHAP explanations by averaging feature importances over multiple training runs.

        The model is randomly initialized and trained on the provided data for each iteration
        using the given trainer and dataloader. SHAP values are computed for the test data
        and aggregated across iterations to obtain stable feature importance scores.

        Args:
            test_data (torch.Tensor): Samples to explain.
            background_data (torch.Tensor): Reference data for SHAP computation.
            nr_iterations (int): Number of iterations to average over.
            trainer: PyTorch Lightning trainer for model fitting.
            dataloader: DataLoader for training the model.

        Returns:
            tuple:
                pd.DataFrame: Aggregated SHAP feature importances.
                dict: Training metrics collected from all iterations.
        """
        dfs = {}
        metrics_dict = {}
        for iteration in range(nr_iterations):
            print(f"Iteration {iteration}")
            self.model.reset_params()
            self.model.init_weights()
            if fast_train:
                optimizer = self.model.configure_optimizers()[0][0]
                self.model, return_dict = self.fast_train(dataloader, max_epochs, optimizer)
                metrics_dict[iteration] = return_dict
            else:
                trainer = pl.Trainer(max_epochs=max_epochs)
                trainer.fit(self.model, dataloader)
            df = self.explain(test_data, background_data)
            dfs[iteration] = df

        col_names = [f"value_{n}" for n in range(len(list(dfs.keys())))]
        values = [df.value.values for df in dfs.values()]
        values = np.array(values)
        values_mean = np.mean(values, axis=0)
        values_std = np.std(values, axis=0)
        df = dfs[0].copy()
        df.drop(columns=["value"], inplace=True)
        df[col_names] = values.T
        df["value_mean"] = values_mean
        df["values_std"] = values_std
        df["value"] = values_mean
        return df, metrics_dict

    def recursive_pathway_elimination(
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
        rpe = RecursivePathwayElimination(self.model, self)
        return_dict = rpe.fit(
            input_data=input_data,
            design_matrix=design_matrix,
            nr_iterations=nr_iterations,
            max_epochs=max_epochs,
            clip_threshold=clip_threshold,
            constant_removal_rate=constant_removal_rate,
            min_features_per_layer=min_features_per_layer,
            early_stopping=early_stopping,
        )

        self.rpe_model = rpe.get_final_model()
        self.rpe_data = rpe.get_final_data()

        return return_dict

    def explain_input(
        self, test_data: torch.Tensor, background_data: torch.Tensor
    ) -> dict:
        """
        Compute SHAP explanations for a specific model layer.

        This method calculates Shapley values for each feature in the given test data
        based on the provided background data, focusing on a specific layer of the model.
        The resulting feature importances are returned as a dictionary, highlighting
        which features most influence the predictions at that layer.

        Args:
            test_data (torch.Tensor): Samples for which explanations are generated.
            background_data (torch.Tensor): Reference data used for SHAP computation.
            layer (int): Index of the model layer to analyze.

        Returns:
            dict: Feature importances as SHAP values for the specified layer.
        """
        explainer = shap.DeepExplainer(self.model, background_data)
        shap_values = explainer.shap_values(test_data)

        shap_dict = {"features": self.model.layer_names[0], "shap_values": shap_values}

        return shap_dict

    def _explain_layers(
        self, background_data: torch.tensor, test_data: torch.tensor
    ) -> dict:
        """
        Compute SHAP explanations for all layers in the model.

        This helper method calculates Shapley values for each feature across all layers
        of the model using the provided background data. The results are aggregated
        into a dictionary, providing a layer-wise view of feature importance and
        highlighting which features drive predictions at each model layer.

        Args:
            background_data (torch.Tensor): Reference data used to compute SHAP values.
            test_data (torch.Tensor): Samples for which explanations are generated.

        Returns:
            dict: Layer-wise SHAP feature importances, with each key representing a layer.
        """
        feature_index = 0

        intermediate_data = test_data

        shap_dict = {"features": [], "shap_values": []}

        for name, layer in self.model.layers.named_children():
            if isinstance(layer, torch.nn.Linear) and (
                "Residual" not in name or "final" in name
            ):

                explainer = shap.DeepExplainer((self.model, layer), background_data)
                shap_values = explainer.shap_values(test_data, check_additivity=False)
                shap_dict["features"].append(self.model.layer_names[feature_index])
                shap_dict["shap_values"].append(shap_values)
                feature_index += 1

                intermediate_data = layer(intermediate_data)
            if (
                isinstance(layer, torch.nn.Tanh)
                or isinstance(layer, torch.nn.ReLU)
                or isinstance(layer, torch.nn.LeakyReLU)
            ):
                intermediate_data = layer(intermediate_data)
        return shap_dict

    def _explain_layer(
        self, background_data: torch.tensor, test_data: torch.tensor, wanted_layer: int
    ) -> dict:
        intermediate_data = test_data

        shap_dict = {"features": [], "shap_values": []}
        layer_index = 0
        for name, layer in self.model.layers.named_children():
            if isinstance(layer, torch.nn.Linear) and (
                "Residual" not in name or "final" in name
            ):
                if layer_index == wanted_layer:
                    explainer = shap.DeepExplainer((self.model, layer), background_data)
                    shap_values = explainer.shap_values(test_data)
                    shap_dict["features"] += self.model.layer_names[wanted_layer]
                    shap_dict["shap_values"] += shap_values
                    return shap_dict
                layer_index += 1
                intermediate_data = layer(intermediate_data)
            if (
                isinstance(layer, torch.nn.Tanh)
                or isinstance(layer, torch.nn.ReLU)
                or isinstance(layer, torch.nn.LeakyReLU)
            ):
                intermediate_data = layer(intermediate_data)
        return shap_dict
