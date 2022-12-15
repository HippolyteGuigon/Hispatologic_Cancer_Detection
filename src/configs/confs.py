import yaml
import os


def load_conf(path: str, include=False) -> yaml:
    """
    The goal of this function is to load the
    params
    Arguments:
        -path: str The path of configuration file to be loaded
        -include: bool: True if the user wants to charge a meta
        file with all configs params
    Returns:
        -file: yaml file The configuration file loaded
    """
    if include:
        with open(path, "r") as f:
            file = yaml.load(f, Loader)

    else:
        with open(path, "r") as ymlfile:
            file = yaml.safe_load(ymlfile)
    return file


class Loader(yaml.SafeLoader):
    """
    The goal of this class is to make it possible to add
    all configs file with yml extension in the same file"""

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super(Loader, self).__init__(stream)

    def include(self, node):

        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, "r") as f:
            return yaml.load(f, Loader)


Loader.add_constructor("!include", Loader.include)
