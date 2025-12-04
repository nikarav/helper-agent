from functools import reduce
from os.path import expandvars

import yaml


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            super().__setitem__(key, self._convert(value))

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            return super().__getattr__(k)
        return v

    def __setattr__(self, k, v):
        self[k] = v

    def __setitem__(self, k, v):
        super().__setitem__(k, self._convert(v))

    def __getitem__(self, k):
        if isinstance(k, str) and "." in k:
            k = k.split(".")
        if isinstance(k, (list, tuple)):
            return reduce(lambda d, kk: d[kk], k, self)
        return super().__getitem__(k)

    def get(self, k, default=None):
        if isinstance(k, str) and "." in k:
            try:
                return self[k]
            except KeyError:
                return default
        return super().get(k, default)

    @staticmethod
    def _convert(value):
        if isinstance(value, dict):
            return DotDict(value)
        if isinstance(value, list):
            return [DotDict._convert(item) for item in value]
        if isinstance(value, tuple):
            return tuple(DotDict._convert(item) for item in value)
        return value


def load_configurations(path: str) -> DotDict:
    """
    Used for parsing configuration files

    :param str path: path to conf file
    :returns DotDict: dictionary accessing fields with dot notation
    """
    with open(path) as f:
        cfg = yaml.safe_load(expandvars(f.read()))
    return DotDict(cfg)


def print_config(config: DotDict, indent: int = 0) -> str:
    """
    Generate a formatted string representation of the configuration.

    :param config: Configuration object to format
    :param indent: Indentation level (default: 0)
    :return: Formatted string representation
    """
    lines = []
    prefix = " " * indent

    for key, value in config.items():
        if isinstance(value, DotDict):
            lines.append(f"{prefix}{key}:")
            lines.append(print_config(value, indent + 2))
        elif isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_print_dict(value, indent + 2))
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            lines.append(_print_sequence(value, indent + 2))
        else:
            lines.append(f"{prefix}{key}: {value}")

    return "\n".join(lines)


def _print_dict(d: dict, indent: int) -> str:
    """
    Helper function to print a dictionary with indentation.

    :param d: Dictionary to format
    :param indent: Indentation level
    :return: Formatted string representation
    """
    lines = []
    prefix = " " * indent

    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_print_dict(value, indent + 2))
        elif isinstance(value, DotDict):
            lines.append(f"{prefix}{key}:")
            lines.append(print_config(value, indent + 2))
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            lines.append(_print_sequence(value, indent + 2))
        else:
            lines.append(f"{prefix}{key}: {value}")

    return "\n".join(lines)


def _print_sequence(seq: list, indent: int) -> str:
    """
    Helper function to print a list with indentation.

    :param seq: Sequence to format
    :param indent: Indentation level
    :return: Formatted string representation
    """
    lines = []
    prefix = " " * indent

    for item in seq:
        if isinstance(item, DotDict):
            lines.append(f"{prefix}-")
            lines.append(print_config(item, indent + 2))
        elif isinstance(item, dict):
            lines.append(f"{prefix}-")
            lines.append(_print_dict(item, indent + 2))
        elif isinstance(item, list):
            lines.append(f"{prefix}-")
            lines.append(_print_sequence(item, indent + 2))
        else:
            lines.append(f"{prefix}- {item}")

    return "\n".join(lines)
