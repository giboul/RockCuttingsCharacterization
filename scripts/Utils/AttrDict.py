import os
import yaml  # pip install pyaml
import json


def load_config(cfg_path):
    """ This function checks the file type and returns the AttrDict object """
    # Extracting file extension
    ext = os.path.splitext(cfg_path)[-1]
    # Checking its type
    if ext == '.json':
        return AttrDict.from_json_path(cfg_path)
    elif ext in ('.yaml', '.yml'):
        return AttrDict.from_yaml_path(cfg_path)
    else:
        raise ValueError(
            f"Unsupported config file format {ext}."
            f"Only '.json', '.yaml' and '.yml' files are supported.")


class AttrDict(dict):
    """
    Dictionary subclass whose entries can be accessed like attributes
    (as well as normally).
    """

    def __init__(self, *args, **kwargs):
        """
        Build a AttrDict from dict like this : AttrDict.from_nested_dicts(dict)
        """
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_json_path(path: str):  # -> AttrDict:
        """ Construct nested AttrDicts from a json. """
        assert os.path.isfile(path), f'Path {path} does not exist.'
        with open(path, 'r') as fn:
            data = json.load(fn)
        return AttrDict.from_nested_dicts(data)

    @staticmethod
    # -> AttrDict:
    def from_yaml_path(path: str, loader: yaml.Loader = yaml.SafeLoader):
        """ Construct nested AttrDicts from a YAML path with the
         specified yaml.loader. """
        if os.path.isfile(path):
            with open(path, 'r') as fn:
                data = yaml.load(fn, Loader=loader)
            return AttrDict.from_nested_dicts(data)
        else:
            raise FileNotFoundError(path)

    def to_yaml(self, path: str):
        """ Save the nested AttrDicts in a YAML file specified by path"""
        assert os.path.isdir(os.path.dirname(
            path)), f'Path {os.path.dirname(path)} does not exist.'
        with open(path, 'w') as fn:
            yaml.dump(self.to_nested_dicts(self), fn, sort_keys=False)

    def to_json(self, path: str):
        """ Save the nested AttrDicts (self) in a JSON
         file specified by path """
        assert os.path.isdir(os.path.dirname(
            path)), f'Path {os.path.dirname(path)} does not exist.'
        with open(path, 'w') as fn:
            json.dump(self.as_json_proof(self), fn)

    @staticmethod
    def from_nested_dicts(data: dict):  # -> AttrDict:
        """ Construct nested AttrDicts from nested dictionaries. """
        if not isinstance(data, dict):
            return data
        else:
            return AttrDict({
                key: AttrDict.from_nested_dicts(data[key])
                for key in data
            })

    @staticmethod
    def to_nested_dicts(data):  # -> AttrDict:
        """ Construct nested dict from an AttrDict. """
        if not isinstance(data, AttrDict):
            return data
        else:
            return dict({
                key: AttrDict.to_nested_dicts(data[key])
                for key in data
            })


if __name__ == '__main__':
    print(load_config('config/raw/template.yaml'))
