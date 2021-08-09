from copy import copy
from singleton import Singleton
import yaml


class Parameters(object):
    __metaclass__ = Singleton

    def __init__(self):
        object.__setattr__(self, "should_hash", {})
        self.should_hash["should_hash"] = False
        pass

    def __setattr__(self, key, val):
        if hasattr(self, key):
            object.__setattr__(self, key, val)
        else:
            raise AttributeError("Attribute " + key + " does not exist.")

    def set(self, key, val):
        """
            Update the value of a parameter. The parameter needs
            to exist already. It is equivalent to
            params.key = val
        """
        if hasattr(self, key):
            setattr(self, key, val)
        else:
            raise Exception('The attribute "{}" does not exist ! '.format(key)
                            + 'You must define it first')

    def define(self, key, val, should_hash=True):
        if not hasattr(self, key):
            object.__setattr__(self, key, val)
            self.should_hash[key] = should_hash
        else:
            raise Exception("The key " + key + " is conflicting with an " +
                            "already existing key ! You might " +
                            "want to set this value instead ?")

    def load(self, filename):
        with open(filename) as fp:
            yaml_params = yaml.load(fp, Loader=yaml.FullLoader)

        if "should_hash" in yaml_params:
            should_hash = yaml_params["should_hash"].copy()
            self.should_hash.update(should_hash)
            del yaml_params["should_hash"]

        for k, v in yaml_params.iteritems():
            self.set(k, v)

    def load_from_parser(self, args):
        for key, val in args.__dict__.iteritems():
            if val and hasattr(self, key):
                if type(self.__dict__[key]) == bool:
                    if val == 'ON':
                        setattr(self, key, True)
                    elif val == 'OFF':
                        setattr(self, key, False)
                else:
                    setattr(self, key, val)

    def define_from_file(self, filename):
        """
            Define and set parameters from a file
            If a parameter already exist, it is updated by the file)

            /!\ if should_hash is not part of the params file,
            it is assumed that everyting should be hashed
        """
        with open(filename) as fp:
            yaml_params = yaml.load(fp, Loader=yaml.FullLoader)

        if "should_hash" in yaml_params:
            should_hash = yaml_params["should_hash"].copy()
            self.should_hash.update(should_hash)
            del yaml_params["should_hash"]
        else:
            for k in yaml_params:
                self.should_hash[k] = True

        for k, v in yaml_params.iteritems():
            object.__setattr__(self, k, v)

    def add_arguments(self, parser):
        for key, val in self.__dict__.iteritems():
            if type(val) == list or type(val) == tuple:
                parser.add_argument('--' + key, nargs='+', type=type(val[0]))
            elif type(val) == dict:
                pass
            elif type(val) == bool:
                parser.add_argument('--' + key, choices = ['ON', 'OFF'], nargs='?')
            else:
                parser.add_argument('--' + key, type=type(val))

    def save(self, filename):
        with open(filename, 'w+') as fp:
            yaml.dump(self.__dict__,
                      fp,
                      default_flow_style=False)

    def get_hash(self):
        dict_to_hash = {}
        for key, val in self.__dict__.iteritems():
            if self.should_hash[key]:
                dict_to_hash[key] = val
        # dict_to_hash = copy(self.__dict__)

        for key, val in dict_to_hash.iteritems():
            if type(val) == list:
                dict_to_hash[key] = tuple(val)
            elif type(val) == dict:
                dict_to_hash[key] = frozenset(val.items())
        return hash(frozenset(dict_to_hash.items()))

    def __repr__(self):
        repr_str = "{\n"
        for k in sorted(self.__dict__):
            if k not in ["should_hash"]:
                repr_str += "\t{}: {}\n".format(k, self.__dict__[k])
        repr_str += "}"
        return repr_str


params = Parameters()
