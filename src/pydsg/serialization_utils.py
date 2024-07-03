import dataclasses
import numpy as np
import shapely.geometry as geo
import shapely
import shapely.wkt
import yaml
import os
from spark_dsg._dsg_bindings import NodeSymbol
import parse


class PolygonList(list):
    pass


class NodeSymbolList(list):
    pass


class DictList(list):
    pass


def yaml_dump(fn, obj):
    with open(fn, "w") as fo:
        yaml.dump(obj, fo)


def yaml_load(fn):
    if not os.path.exists(fn):
        return None
    with open(fn, "r") as fo:
        y = yaml.safe_load(fo)
    return y


def np_save(fn, obj):
    if type(obj) != np.ndarray:
        np.save(fn, np.array(obj, dtype=object), allow_pickle=True)
    else:
        np.save(fn, obj)


def np_load(fn):
    if not os.path.exists(fn):
        return None
    obj = np.load(fn, allow_pickle=True)
    if obj.dtype == "O":
        return obj.tolist()
    else:
        return obj


# def node_symbols_save(fn, obj):
#    str_lines = [str(s) for s in obj]
#    with open(fn, "w") as fo:
#        fo.writelines(str_lines)


def node_symbols_load(fn):
    if not os.path.exists(fn):
        return None
    with open(fn, "r") as fo:
        strings = fo.readlines()

    symbols = [
        NodeSymbol(*parse.parse("{char}({index})", s).named.keys) for s in strings
    ]
    return symbols


def list_save(fn, obj):
    str_lines = "\n".join([str(s) for s in obj])
    with open(fn, "w") as fo:
        fo.writelines(str_lines)


def list_load(fn):
    if not os.path.exists(fn):
        return None

    with open(fn, "r") as fo:
        lines = fo.readlines()

    return [s.replace("\n", "") for s in lines]


def poly_save(fn, polys):
    """Save a list of shapely polygons as WKT"""
    with open(fn, "w") as fo:
        for p in polys:
            wkt = p.wkt
            fo.write(wkt)
            fo.write("\n")


def poly_load(fn):
    """Load a file where each line is a WKT to list of shapely polygons"""
    if not os.path.exists(fn):
        return None
    with open(fn, "r") as fo:
        wkt_lines = fo.readlines()
    return list(map(shapely.wkt.loads, wkt_lines))


class Serializable:

    def save(self, base_path):
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        for field in dataclasses.fields(self):
            print(f"Saving field {field.name} of type {field.type}")
            name = field.name
            if name in self.get_unserializable_fields():
                continue
            val = getattr(self, name)
            typ = field.type
            path = os.path.join(base_path, name)
            if issubclass(typ, dict):
                yaml_dump(path + ".yaml", val)
            elif issubclass(typ, np.ndarray):
                np_save(path + ".npy", val)
            elif issubclass(typ, geo.Polygon):
                poly_save(path + ".wkt", [val])
            elif issubclass(typ, PolygonList):
                poly_save(path + ".wkt", val)
            elif issubclass(typ, DictList):
                yaml_dump(path + ".yaml", val)
            elif issubclass(typ, list):
                list_save(path + ".lst", val)
            elif val is None:
                print(f"Warning: Field {name} is None. Not serializing")
            else:
                try:
                    val.save(path)
                except AttributeError as e:
                    print(e)
                    raise Exception(
                        f"Cannot serialize field {name} of type {type(val)}"
                    )

    @classmethod
    def load(cls, base_path):
        args = {}
        for field in dataclasses.fields(cls):
            name = field.name
            typ = field.type
            path = os.path.join(base_path, name)
            if typ == dict:
                args[name] = yaml_load(path + ".yaml")
            elif typ == np.ndarray:
                args[name] = np_load(path + ".npy")
            elif typ == geo.Polygon:
                args[name] = poly_load(path + ".wkt")[0]
            elif typ == PolygonList:
                args[name] = poly_load(path + ".wkt")
            elif typ == DictList:
                args[name] = yaml_load(path + ".yaml")
            elif typ == list:
                args[name] = list_load(path + ".lst")
            else:
                try:
                    args[name] = typ.load(path)
                except AttributeError as e:
                    print(e)

        return cls(**args)
