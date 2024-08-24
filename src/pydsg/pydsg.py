from dataclasses import dataclass
import dataclasses
import numpy as np
import shapely.geometry as geo
from pydsg.serialization_utils import (
    PolygonList,
    DictList,
    Serializable,
)
from pydsg.soa_aos_utils import SoAIndexing, StructAppend
import copy
import os


@dataclass
class DsgNode:
    hydra_symbol: str
    from_hydra: bool
    center: np.ndarray
    semantic_label: str
    semantic_color: np.ndarray


def remove_extra_str_from_edge_map(symbols, dictionary):
    return {
        s: [d for d in dictionary[s] if d in symbols]
        for s in dictionary.keys()
        if s in symbols
    }


@dataclass
class DsgLayer(Serializable, SoAIndexing, StructAppend):

    hydra_symbol: list
    from_hydra: np.ndarray
    center: np.ndarray
    semantic_label: list
    semantic_color: np.ndarray

    hydra_index_to_local_index: dict
    local_index_to_hydra_index: dict
    parent_dict: dict
    sibling_dict: dict
    children_dict: dict

    def __len__(self):
        return len(self.hydra_symbol)

    def get_from_symbol(self, symbol):
        return self[self.hydra_index_to_local_index[symbol]]

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(self.hydra_index_to_local_index[key])
        elif isinstance(key, list) and isinstance(key[0], str):
            return super().__getitem__(
                [self.hydra_index_to_local_index[k] for k in key]
            )
        else:
            return super().__getitem__(key)

    def get_unserializable_fields(self):
        return ["element_struct_type"]

    def update_local_hydra_mapping(self):
        self.hydra_index_to_local_index = {
            s: ix for ix, s in enumerate(self.hydra_symbol)
        }
        self.local_index_to_hydra_index = {
            ix: s for ix, s in enumerate(self.hydra_symbol)
        }

    def cleanup(self, existing_symbols_list=None):
        # remove nodes for the edge dict that no longer exist in hydra_symbol list
        if existing_symbols_list is None:
            existing_symbols_list = self.hydra_symbol
        self.update_local_hydra_mapping()
        self.parent_dict = remove_extra_str_from_edge_map(
            existing_symbols_list, self.parent_dict
        )
        self.sibling_dict = remove_extra_str_from_edge_map(
            existing_symbols_list, self.sibling_dict
        )
        self.children_dict = remove_extra_str_from_edge_map(
            existing_symbols_list, self.children_dict
        )

        # ensure all symbols are keys in the connectivity dicts
        for s in self.hydra_symbol:
            if s not in self.parent_dict:
                self.parent_dict[s] = []
            if s not in self.sibling_dict:
                self.sibling_dict[s] = []
            if s not in self.children_dict:
                self.children_dict[s] = []


@dataclass
class Object(DsgNode):
    """Class for individual object"""

    orientation: np.ndarray
    movable: bool
    surface: bool
    box: np.ndarray
    boundary_shapely: geo.Polygon


@dataclass
class Place2d(DsgNode):
    """Class for individual 2d place"""

    boundary: np.ndarray
    boundary_shapely: geo.Polygon
    original_area: float


@dataclass
class Place3d(DsgNode):
    """Class for individual 3d place"""

    boundary: np.ndarray
    radius: float
    predicted_place: bool
    frontier: bool


@dataclass
class Room(DsgNode):
    """Class for individual room"""

    box: np.ndarray
    boundary_shapely: geo.Polygon
    label_confidences: dict
    predicted: bool


@dataclass
class Building(DsgNode):
    """Class for individual building"""

    # box: np.ndarray
    # boundary_shapely: geo.Polygon


@dataclass
class ObjectLayer(DsgLayer):
    """Class representing all objects"""

    orientation: np.ndarray
    movable: np.ndarray
    surface: np.ndarray
    box: np.ndarray
    boundary_shapely: PolygonList

    element_struct_type: type = Object


@dataclass
class Place2dLayer(DsgLayer):
    """Class representing all 2d places"""

    boundary: np.ndarray
    boundary_shapely: PolygonList
    original_area: np.ndarray

    element_struct_type: type = Place2d


@dataclass
class Place3dLayer(DsgLayer):
    """Class representing all 3d places"""

    boundary: np.ndarray
    radius: np.ndarray
    predicted_place: np.ndarray
    frontier: np.ndarray

    element_struct_type: type = Place3d


@dataclass
class RoomLayer(DsgLayer):
    """Class representing all rooms"""

    box: np.ndarray
    boundary_shapely: PolygonList
    label_confidences: DictList
    predicted: np.ndarray
    edge_probability_dict: dict
    edge_probabilities: np.ndarray = None
    # connection_probability: np.ndarray = None

    element_struct_type: type = Room


@dataclass
class BuildingLayer(DsgLayer):
    """Class representing all buildings"""

    # box: np.ndarray
    # boundary_shapely: PolygonList

    element_struct_type: type = Building


@dataclass
class PyDSG(Serializable):
    """Class representing full scene graph"""

    objects: ObjectLayer = None
    places_2d: Place2dLayer = None
    places_3d: Place3dLayer = None
    rooms: RoomLayer = None
    buildings: BuildingLayer = None

    gtsam_symbol_to_index: dict = None
    index_to_hydra_symbol: dict = None

    def __copy__(self):
        copy_dict = {
            f.name: copy.copy(getattr(self, f.name)) for f in dataclasses.fields(self)
        }
        new = type(self)(**copy_dict)
        return new

    @classmethod
    def load(cls, base_path):
        if not os.path.exists(base_path):
            raise Exception(f"Could not load scene graph. No file {base_path}.")
        return super().load(base_path)

    def get_unserializable_fields(self):
        fields = []

        if self.objects is not None:
            fields += self.objects.get_unserializable_fields()
        if self.places_2d is not None:
            fields += self.places_2d.get_unserializable_fields()
        if self.places_3d is not None:
            fields += self.places_3d.get_unserializable_fields()
        if self.rooms is not None:
            fields += self.rooms.get_unserializable_fields()
        if self.buildings is not None:
            fields += self.buildings.get_unserializable_fields()
        return fields

    def cleanup(self):

        full_symbol_list = []
        if self.objects is not None:
            full_symbol_list += self.objects.hydra_symbol
        if self.places_2d is not None:
            full_symbol_list += self.places_2d.hydra_symbol
        if self.places_3d is not None:
            full_symbol_list += self.places_3d.hydra_symbol
        if self.rooms is not None:
            full_symbol_list += self.rooms.hydra_symbol
        if self.buildings is not None:
            full_symbol_list += self.buildings.hydra_symbol

        if self.objects is not None:
            self.objects.cleanup(full_symbol_list)
        if self.places_2d is not None:
            self.places_2d.cleanup(full_symbol_list)
        if self.places_3d is not None:
            self.places_3d.cleanup(full_symbol_list)
        if self.rooms is not None:
            self.rooms.cleanup(full_symbol_list)
        if self.buildings is not None:
            self.buildings.cleanup(full_symbol_list)
