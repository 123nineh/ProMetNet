import os
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import plotly.graph_objects as go
import matplotlib.colors as mcolors

def subgraph_sankey(
    df: pd.DataFrame,
    final_node: int = 0,
    val_col="value",
    cmap_name="coolwarm",
):
    """
    Generate a Sankey diagram for visualizing pathway flows.

    This method creates a Plotly Sankey diagram from the input data,
    highlighting the flow of values between nodes for hierarchical network analysis.

    Args:
        df (pd.DataFrame): Input data containing source, target, and value information.
        final_node (str, optional): Node to use as the final target in the diagram. Default is "root".
        val_col (str, optional): Column representing flow values. Default is "value".
        cmap_name (str, optional): Color map name for the diagram. Default is "coolwarm".

    Returns:
        go.Figure: Plotly Figure object representing the Sankey diagram.
    """
    df["source layer"] = df["source layer"].astype(int)
    df["target layer"] = df["target layer"].astype(int)
    unique_features = df["source"].unique().tolist()
    unique_features += df["target"].unique().tolist()
    code_map, feature_labels = _encode_features(list(set(unique_features)))
    sources = df["source"].unique().tolist()
    name_map = _create_subgraph_name_map(df)

    def normalize_layer_values(df: pd.DataFrame):
        new_df = pd.DataFrame()
        total_value_sum = df[val_col].sum()
        for layer in df["source layer"].unique():
            layer_df = df[df["source layer"] == layer].copy()
            layer_total = layer_df[val_col].sum()
            layer_df.loc[:, "normalized value"] = (
                total_value_sum * layer_df[val_col] / layer_total
            )
            new_df = pd.concat([new_df, layer_df])
        return new_df

    df = _remove_loops(df)
    df = normalize_layer_values(df)

    def get_connections(sources: list, source_target_df: pd.DataFrame):
        conn = source_target_df[source_target_df["source"].isin(sources)]
        source_code = [_get_code(s, code_map) for s in conn["source"]]
        target_code = [_get_code(s, code_map) for s in conn["target"]]
        values = [v for v in conn["normalized value"]]
        link_colors = conn["node_color"].values.tolist()
        return source_code, target_code, values, link_colors

    def get_node_colors(sources, df: pd.DataFrame):
        cmap = plt.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=0, vmax=1), cmap=cmap_name
        )
        new_df = df
        node_dict = {}
        weight_dict = {}
        for layer in df["source layer"].unique():
            w = df.loc[df["source layer"] == layer, "normalized value"].values
            n = df.loc[df["source layer"] == layer, "source"].values

            if len(w) < 2:
                w = [1]
            else:
                xmin = min(w)
                xmax = max(w)
                if xmax == xmin:
                    w = [1] * len(n)
                else:
                    w = np.array(w)
                    X_std = (w - xmin) / (xmax - xmin)
                    X_scaled = X_std
                    w = X_scaled.tolist()

            pairs = [(f, v) for f, v in zip(n, w)]
            for pair in pairs:
                n, w = pair
                r, g, b, a = cmap.to_rgba(w, alpha=0.5)
                weight_dict[n] = w
                node_dict[n] = f"rgba({r * 255}, {g * 255}, {b * 255}, {a})"
        node_dict[final_node] = "rgba(0,0,0,1)"
        weight_dict[final_node] = 1
        colors = [node_dict[n] for n in sources]
        new_df = new_df.assign(
            node_color=[node_dict[n] for n in new_df["source"]],
            node_weight=[weight_dict[n] for n in new_df["source"]],
        )
        return new_df, colors

    df, node_colors = get_node_colors(feature_labels, df)
    encoded_source, encoded_target, value, link_colors = get_connections(sources, df)

    new_labels = []
    for label in feature_labels:
        if label in name_map.keys():
            new_labels.append(name_map[label])
        else:
            new_labels.append(label)

    fig = _plot_layered_network(
        df=df,
        source_col="source",
        target_col="target",
        source_layer_col="source layer",
        target_layer_col="target layer",
        value_col="normalized value",
        label_map=dict(zip(feature_labels, new_labels)),
        final_node=final_node,
        node_order=feature_labels,
        node_colors=node_colors,
        link_colors=link_colors,
        cmap_name=cmap_name,
    )

    return fig


def complete_sankey(
    df: pd.DataFrame,
    multiclass: bool = False,
    show_top_n: int = 10,
    val_col: str = "value",
    node_cmap: str = "Reds",
    edge_cmap: Union[str, list] = "coolwarm",
    root_id: int = 0,
    other_id: int = -1,
    save_file: str = None
):
    if not multiclass:
        df = df.groupby(
            by=["source", "target", "source name", "target name"], as_index=False
        ).agg(
            {
                "value": "sum",
                "source layer": "mean",
                "target layer": "mean",
                "type": "mean",
            }
        )

    df["source layer"] = df["source layer"].astype(int)
    df["target layer"] = df["target layer"].astype(int)

    df = _remove_loops(df)
    n_layers = max(df["target layer"].values)
    df["value"] = df[val_col]

    name_map = _create_name_map(df, n_layers, root_id, other_id)

    top_n = {}

    for layer in range(n_layers):
        top_n[layer] = (
            df.loc[df["source layer"] == layer]
            .groupby("source")
            .mean(numeric_only=True)
            .sort_values("value", ascending=False)
            .iloc[:show_top_n]
            .index.tolist()
        )

    def set_to_other(row, top_n: dict, source_or_target: str):
        node = row[source_or_target]
        layer = row["source layer"] + 1
        if source_or_target == "target":
            layer = layer + 1
        for top_layer_values in top_n.values():
            if node in top_layer_values:
                return node
        if node == root_id:
            return node
        return other_id * layer  # All other nodes are multiples of other_id

    def normalize_layer_values(df: pd.DataFrame):
        new_df = pd.DataFrame()
        df["Other"] = (
            df["source_w_other"]
            .apply(lambda x: True if x <= other_id else False)
            .copy()
        )
        other_df = df[df["Other"] == True]
        df = df[df["Other"] == False]
        for layer in df["source layer"].unique():
            layer_df = df[df["source layer"] == layer].copy()
            layer_total = layer_df["value"].sum()
            layer_df["normalized value"] = layer_df["value"] / layer_total
            new_df = pd.concat([new_df, layer_df])
        for layer in other_df["source layer"].unique():
            layer_df = other_df[other_df["source layer"] == layer].copy()
            layer_total = layer_df["value"].sum()
            layer_df["normalized value"] = 0.1 * layer_df["value"] / layer_total
            new_df = pd.concat([new_df, layer_df])
        return new_df

    def get_connections(sources: list, df: pd.DataFrame):
        conn = df[df["source_w_other"].isin(sources)].copy()
        source_code = [_get_code(s, code_map) for s in conn["source_w_other"]]
        target_code = [_get_code(s, code_map) for s in conn["target_w_other"]]
        values = [v for v in conn["normalized value"]] * 10
        if multiclass == False:
            temp_df, _ = get_node_colors(feature_labels, df, curr_cmap=edge_cmap)
            link_colors = (
                temp_df["node_color"]
                .apply(lambda x: adjust_opacity(x, 0.8))  # 调整透明度 0.3
                .values.tolist()
            )
        else:
            temp_df, _ = get_node_colors(feature_labels, df, curr_cmap=edge_cmap)
            link_colors = conn.apply(
                lambda x: "rgba(236,236,236, 0.5)" #0.15
                if x["source_w_other"] <= other_id
                else adjust_opacity(
                    temp_df.loc[temp_df["source_w_other"] == x["source_w_other"], "node_color"].values[0], 0.5),
                axis=1,
            ).values.tolist()

        return source_code, target_code, values, link_colors

    def adjust_opacity(rgba_color, alpha_factor):
        rgba_values = rgba_color.strip("rgba()").split(",")
        r, g, b, a = map(float, rgba_values)
        new_alpha = a * alpha_factor
        return f"rgba({int(r)}, {int(g)}, {int(b)}, {new_alpha})"

    def get_node_colors(sources: list, df: pd.DataFrame, curr_cmap=node_cmap):
        cmaps = {}
        for layer in df["source layer"].unique():
            c_df = df[df["source layer"] == layer].copy()
            c_df = c_df[~c_df["source_w_other"] <= other_id]

            cmap = plt.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(
                    vmin=c_df.groupby("source_w_other")
                    .mean(numeric_only=True)["normalized value"]
                    .min()
                    * 0.8,
                    vmax=c_df.groupby("source_w_other")
                    .mean(numeric_only=True)["normalized value"]
                    .max(),
                ),
                cmap=curr_cmap,
            )
            cmaps[layer] = cmap
        colors = []

        new_df = pd.DataFrame()
        for source in sources:
            source_df = df[df["source_w_other"] == source].copy()
            if source <= other_id:
                colors.append("rgb(236,236,236, 0.75)")
                source_df["node_color"] = "rgb(236,236,236, 0.75)"
            elif source == root_id:
                colors.append("rgba(0,0,0,1)")
                source_df["node_color"] = "rgb(0,0,0,1)"
            else:
                intensity = (
                    source_df.groupby("source_w_other")
                    .mean(numeric_only=True)["normalized value"]
                    .values[0]
                )
                cmap = cmaps[source_df["source layer"].unique()[0]]
                r, g, b, a = cmap.to_rgba(intensity, alpha=0.9)
                colors.append(f"rgba({r * 255}, {g * 255}, {b * 255}, {a})")
                source_df["node_color"] = f"rgba({r * 255}, {g * 255}, {b * 255}, {a})"
            new_df = pd.concat([new_df, source_df])
        return new_df, colors

    def get_node_positions(feature_labels: list, df: pd.DataFrame):
        x = []
        y = []
        grouped_df = df.groupby("source_w_other", as_index=False).agg(
            {"source layer": "min", "value": "mean"}
        )
        layers = range(n_layers)
        final_df = pd.DataFrame()
        for layer in layers:
            layer_df = (
                grouped_df[grouped_df["source layer"] == layer]
                .sort_values(["value"], ascending=True)
                .copy()
            )
            other_df = layer_df[layer_df["source_w_other"] <= other_id].copy()
            layer_df = layer_df[layer_df["source_w_other"] > other_id].copy()

            other_value = other_df["value"]

            layer_df["rank"] = range(len(layer_df.index))
            layer_df["value"] = layer_df["value"] / layer_df["value"].sum()

            layer_df["y"] = (
                0.8
                * (0.01 + max(layer_df["rank"]) - layer_df["rank"])
                / (max(layer_df["rank"]))
                - 0.05
            )
            layer_df["x"] = (0.01 + layer) / (len(layers) + 1)
            other_df = pd.DataFrame(
                [
                    [
                        other_id * (layer + 1),
                        layer,
                        other_value,
                        10,
                        0.9,
                        (0.01 + layer) / (len(layers) + 1),
                    ]
                ],
                columns=["source_w_other", "source layer", "value", "rank", "y", "x"],
            )

            final_df = pd.concat([final_df, layer_df, other_df])

        for f in feature_labels:
            if f == root_id:
                x.append(0.85)
                y.append(0.5)
            else:
                x.append(final_df[final_df["source_w_other"] == f]["x"].values)
                y.append(final_df[final_df["source_w_other"] == f]["y"].values)
        return x, y

    df["source_w_other"] = df.apply(lambda x: set_to_other(x, top_n, "source"), axis=1)
    df["target_w_other"] = df.apply(lambda x: set_to_other(x, top_n, "target"), axis=1)
    df = df[
        ~((df["source_w_other"] <= other_id) & (df["target_w_other"] <= other_id))
    ].copy()
    df = normalize_layer_values(df)
    df = df.groupby(
        by=["source_w_other", "target_w_other", "type"], sort=False, as_index=False
    ).agg(
        {
            "normalized value": "sum",
            "value": "sum",
            "source layer": "mean",
            "target layer": "mean",
        }
    )
    unique_features = (
        df["source_w_other"].unique().tolist() + df["target_w_other"].unique().tolist()
    )
    code_map, feature_labels = _encode_features(list(set(unique_features)))
    sources = df["source_w_other"].unique().tolist()

    df, node_colors = get_node_colors(feature_labels, df)
    encoded_source, encoded_target, value, link_colors = get_connections(sources, df)

    x, y = get_node_positions(feature_labels, df)

    new_labels = []
    for label in feature_labels:
        if label in name_map.keys():
            new_labels.append(name_map[label])
        else:
            new_labels.append(label)

    fig = _plot_layered_network(
        df=df,
        source_col="source_w_other",
        target_col="target_w_other",
        source_layer_col="source layer",
        target_layer_col="target layer",
        value_col="normalized value",
        label_map=dict(zip(feature_labels, new_labels)),
        final_node=root_id,
        node_order=feature_labels,
        node_colors=node_colors,
        link_colors=link_colors,
        original_x=x,
        original_y=y,
        cmap_name=node_cmap,
    )
    return fig


# ----------------------------------------------------------------


def _plot_layered_network(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    source_layer_col: str,
    target_layer_col: str,
    value_col: str,
    label_map: dict,
    final_node=None,
    node_order=None,
    node_colors=None,
    link_colors=None,
    original_x=None,
    original_y=None,
    cmap_name="Accent",
):
    source_nodes = df[source_col].drop_duplicates().tolist()
    target_nodes = df[target_col].drop_duplicates().tolist()
    all_nodes = []
    if node_order is not None:
        for node in node_order:
            if (node in source_nodes or node in target_nodes or node == final_node) and node not in all_nodes:
                all_nodes.append(node)
    for node in source_nodes + target_nodes:
        if node not in all_nodes:
            all_nodes.append(node)
    if final_node is not None and final_node not in all_nodes:
        all_nodes.append(final_node)

    layer_map = {}
    for node in all_nodes:
        source_rows = df[df[source_col] == node]
        target_rows = df[df[target_col] == node]
        if len(source_rows) > 0:
            layer_map[node] = int(source_rows[source_layer_col].min())
        elif len(target_rows) > 0:
            layer_map[node] = int(target_rows[target_layer_col].min())
        else:
            layer_map[node] = 0
    if final_node is not None:
        layer_map[final_node] = max(layer_map.values()) + 1

    position_map = {}
    if original_x is not None and original_y is not None and node_order is not None:
        for node, px, py in zip(node_order, original_x, original_y):
            try:
                px, py = float(px), float(py)
                if np.isfinite(px) and np.isfinite(py):
                    position_map[node] = (px, py)
            except (TypeError, ValueError):
                pass

    layers = sorted(set(layer_map.values()))
    for layer in layers:
        layer_nodes = [n for n in all_nodes if layer_map[n] == layer]
        missing_nodes = [n for n in layer_nodes if n not in position_map]
        if len(layers) <= 1:
            x_pos = 0.5
        else:
            x_pos = 0.08 + 0.76 * layer / max(1, len(layers) - 1)
        normal_nodes = [n for n in missing_nodes if not _is_other_node(n)]
        other_nodes = [n for n in missing_nodes if _is_other_node(n)]
        if len(normal_nodes) == 1:
            normal_y = [0.5]
        elif len(normal_nodes) > 1:
            normal_y = np.linspace(0.08, 0.82, len(normal_nodes)).tolist()
        else:
            normal_y = []
        for node, yy in zip(normal_nodes, normal_y):
            position_map[node] = (x_pos, yy)
        for node in other_nodes:
            position_map[node] = (x_pos, 0.92)

    if final_node is not None:
        position_map[final_node] = (0.94, 0.5)

    if node_colors is None:
        node_color_map = {}
    else:
        node_color_map = {node: color for node, color in zip(node_order or [], node_colors)}

    fig = go.Figure()


    for i, (_, row) in enumerate(df.iterrows()):
        source = row[source_col]
        target = row[target_col]
        if source not in position_map or target not in position_map:
            continue
        x1, y1 = position_map[source]
        x2, y2 = position_map[target]
        edge_color = link_colors[i] if link_colors is not None and i < len(link_colors) else "rgba(80,80,80,0.45)"
        try:
            if edge_color.startswith("rgba"):
                parts = edge_color.strip("rgba()").split(",")
                parts = [p.strip() for p in parts]
                if len(parts) == 4:
                    parts[3] = str(float(parts[3]) * 0.7)
                    edge_color = "rgba(" + ",".join(parts) + ")"
        except Exception:
            pass
        fig.add_trace(
            go.Scatter(
                x=[x1, x2], y=[y1, y2], mode="lines",
                line=dict(color=edge_color, width=1.5),
                hoverinfo="skip", showlegend=False,
            )
        )
        fig.add_annotation(
            x=x2, y=y2, ax=x1, ay=y1,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=0.8,
            arrowwidth=1.3, arrowcolor=edge_color, text="",
        )

    for node in all_nodes:
        if node not in position_map:
            continue
        x_pos, y_pos = position_map[node]
        if final_node is not None and node == final_node:
            node_color = "rgba(0,0,0,1)"
        elif node in node_color_map:
            node_color = node_color_map[node]
        elif _is_other_node(node):
            node_color = "rgba(236,236,236,0.75)"
        else:
            node_color = "rgba(120,120,120,1)"
        label = label_map.get(node, str(node))

        half_w, half_h = 0.008, 0.008
        fig.add_shape(
            type="rect",
            x0=x_pos - half_w, x1=x_pos + half_w,
            y0=y_pos - half_h, y1=y_pos + half_h,
            xref="x", yref="y",
            line=dict(color=node_color, width=1.2),
            fillcolor=node_color, layer="above",
        )
        fig.add_annotation(
            x=x_pos + 0.015, y=y_pos, xref="x", yref="y",
            text=str(label), showarrow=False,
            font=dict(size=14, color="black", family="Arial"),
            xanchor="left", yanchor="middle",
        )

    for layer in layers:
        layer_nodes = [n for n in all_nodes if layer_map[n] == layer]
        xs = [position_map[n][0] for n in layer_nodes if n in position_map]
        if xs:
            fig.add_annotation(
                x=float(np.mean(xs)), y=1.01,
                xref="x", yref="y", text=f"Layer {layer + 1}",
                showarrow=False, font=dict(size=14, color="black", family="Arial"),
                xanchor="center", yanchor="bottom",
            )

    n_layers_draw = max(1, len(layers))
    max_nodes_draw = max(
        [sum(1 for n in all_nodes if layer_map[n] == layer) for layer in layers] or [1]
    )

    fig_width = max(900, 240 * n_layers_draw)
    fig_height = max(650, 105 * max_nodes_draw + 180)

    fig.update_xaxes(
        range=[-0.04, 1.04],
        visible=False,
        fixedrange=False,
        constrain="domain",
    )
    fig.update_yaxes(
        range=[1.10, -0.06],
        visible=False,
        fixedrange=False,
        constrain="domain",
    )
    fig.update_layout(
        width=fig_width,
        height=fig_height,
        autosize=False,
        margin=dict(l=90, r=90, t=95, b=70),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode=False,
        showlegend=False,
        dragmode="pan",
    )
    return fig




def _is_other_node(node):
    try:
        return isinstance(node, (int, np.integer)) and node < 0
    except Exception:
        return False


def _encode_features(features: list):
    feature_map = {"feature": features, "code": list(range(len(features)))}
    feature_map = pd.DataFrame(data=feature_map)
    return feature_map, features


def _get_code(feature: list, feature_map: pd.DataFrame):
    code = feature_map[feature_map["feature"] == feature]["code"].values[0]
    return code


def _remove_loops(df: pd.DataFrame):
    df.loc[:, "loop"] = df.apply(lambda x: x["source name"] == x["target name"], axis=1)
    df = df[df["loop"] == False].copy()
    return df


def _create_name_map(df: pd.DataFrame, n_layers: int, root_id: int, other_id: int):
    name_map = dict(
        zip(df["source"].values.tolist(), df["source name"].values.tolist())
    )
    name_map[root_id] = "Output"
    for i in range(1, n_layers + 1):
        name_map[other_id * i] = f"Other connections {i}"
    return name_map


def _create_subgraph_name_map(df: pd.DataFrame):
    name_map = dict(
        zip(df["source"].values.tolist(), df["source name"].values.tolist())
    )
    name_map.update(
        zip(df["target"].values.tolist(), df["target name"].values.tolist())
    )
    return name_map


