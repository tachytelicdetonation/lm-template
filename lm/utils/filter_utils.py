"""Utils for filtering through the model structure"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
from jax._src.lib import pytree
from jaxtyping import PyTree


@dataclass
class SpecNode:
    @staticmethod
    def from_string(string):
        if string == "**":
            return DoubleWildNode()
        elif string == "*":
            return WildNode()
        elif string.isdigit():
            return IndexNode(int(string))
        else:
            return StringNode(string)

    @staticmethod
    def parse_spec_str(spec: str) -> list[SpecNode]:
        return [SpecNode.from_string(string) for string in spec.split(".")]


@dataclass
class StringNode(SpecNode):
    value: str


@dataclass
class WildNode(SpecNode):
    pass


@dataclass
class DoubleWildNode(SpecNode):
    """Arbitrary length wildcard either for prefix or midfix"""

    pass


@dataclass
class IndexNode(SpecNode):
    index: int

    def __post_init__(self):
        assert self.index >= 0, "Negative indices are not allowed"


def matches(spec: list[SpecNode], path: list[pytree.GetAttrKey | pytree.SequenceKey | pytree.DictKey]) -> bool:
    match spec, path:
        case [[], []]:
            return True
        case [[WildNode(), *s_rest], [_p_cur, *p_rest]]:
            return matches(s_rest, p_rest)
        case [[DoubleWildNode(), *s_rest], [_p_cur, *p_rest]]:
            return matches(s_rest, p_rest) or matches(spec, p_rest)
        case [[IndexNode(n_i), *s_rest], [pytree.SequenceKey(s_i), *p_rest]]:
            return n_i == s_i and matches(s_rest, p_rest)
        case [[StringNode(n_s), *s_rest], [pytree.GetAttrKey(s_s) | pytree.DictKey(s_s), *p_rest]]:
            return n_s == s_s and matches(s_rest, p_rest)
        case _:
            return False


@dataclass
class Spec:
    is_exclude: bool
    spec_parts: list[SpecNode]

    @classmethod
    def from_string(cls, s: str):
        is_exclude = False
        exclude_str = "exclude "
        if s.startswith(exclude_str):
            s = s[len(exclude_str) :]
            is_exclude = True

        return cls(is_exclude=is_exclude, spec_parts=SpecNode.parse_spec_str(s))


@dataclass
class SpecMatch:
    exclude: bool
    match: bool


def reduce_spec(spec_matches: list[SpecMatch]):
    current = False
    for spec_match in spec_matches:
        if spec_match.match:
            if not spec_match.exclude:
                current = True
            else:
                current = False

    return current


def get_filter_spec(tree: PyTree, spec_strs: list[str], filter_type: str):
    """Return a pytree that specifies True for selected params."""
    specs = [Spec.from_string(spec_str) for spec_str in spec_strs]

    specs_matches = [
        jax.tree.map_with_path(lambda path, _value: SpecMatch(exclude=spec.is_exclude, match=matches(spec.spec_parts, path)), tree) for spec in specs
    ]

    for spec_str, spec_matches in zip(spec_strs, specs_matches):
        if "index" not in spec_str:
            assert jax.tree.reduce(lambda a, b: a or b, jax.tree.map(lambda n: n.match, spec_matches)), (
                f"Spec {filter_type} did not match any parameters: {spec_str}"
            )

    selected_params_filter_spec = jax.tree.map(lambda *entries: reduce_spec(entries), *specs_matches)

    return selected_params_filter_spec


def filter_parameters(tree: PyTree, spec_strs: list[str], filter_type: str) -> PyTree:
    """Filters to only include the parameters that match the provided specs."""

    selected_params_filter_spec = get_filter_spec(tree, spec_strs, filter_type)

    selected_params = eqx.filter(tree, selected_params_filter_spec)

    return selected_params


def _dict_flatten(d: dict) -> list[tuple[list[pytree.GetAttrKey | pytree.SequenceKey | pytree.DictKey], Any]]:
    """Flatten a nested dictionary to a list of (path, value) tuples"""

    def flatten_gen(d):
        if isinstance(d, dict):
            for k, v in d.items():
                for path, value in flatten_gen(v):
                    yield [k, *path], value
        else:
            yield [], d

    return list(flatten_gen(d))


def _reduce_to_prefix_paths(tree: PyTree) -> list[tuple[list[pytree.GetAttrKey | pytree.SequenceKey | pytree.DictKey], Any]]:
    """Reduces a pytree to the minimum prefix tree that can be used to represent it."""

    def reduce_tree(tree):
        if not isinstance(tree, dict):
            return tree
        reduced = {k: reduce_tree(v) for k, v in tree.items()}
        assert len(reduced) > 0
        first = reduced[next(iter(reduced))]
        if not isinstance(first, dict) and all(v == first for v in reduced.values()):
            return first
        else:
            return reduced

    tree_from_path = {}
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        tree_ptr = tree_from_path
        for p in path[:-1]:
            if p not in tree_ptr:
                tree_ptr[p] = {}
            tree_ptr = tree_ptr[p]
        tree_ptr[path[-1]] = value

    reduced_tree = reduce_tree(tree_from_path)
    return _dict_flatten(reduced_tree)


def filter_apply_updates(model, updates):
    model = jax.tree.map(lambda p, u: p + u if u is not None else p, model, updates)
    return model


def tree_path_to_string(path, sep=None):
    keys = []
    for key in path:
        if isinstance(key, jax.tree_util.SequenceKey):
            keys.append(str(key.idx))
        elif isinstance(key, jax.tree_util.DictKey):
            keys.append(str(key.key))
        elif isinstance(key, jax.tree_util.GetAttrKey):
            keys.append(str(key.name))
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            keys.append(str(key.key))
        else:
            keys.append(str(key))
    if sep is None:
        return tuple(keys)
    return sep.join(keys)


def get_mask_fn(match_name_fn, params):
    mask = jax.tree_util.tree_map_with_path(lambda path, _: match_name_fn(tree_path_to_string(path, sep="/")), params)
    return mask
