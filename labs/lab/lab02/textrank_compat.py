"""Compatibility helpers for running textrank4zh on newer Python environments."""

import networkx as nx


def patch_networkx_for_textrank4zh() -> None:
    if hasattr(nx, "from_numpy_matrix"):
        return
    nx.from_numpy_matrix = nx.from_numpy_array