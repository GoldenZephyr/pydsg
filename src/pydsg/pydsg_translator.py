import spark_dsg

from pydsg.soa_aos_utils import to_soa
from pydsg.pydsg import (
    PyDSG,
    Object,
    Place2d,
    Place3d,
    Room,
    Building,
    ObjectLayer,
    Place2dLayer,
    Place3dLayer,
    RoomLayer,
    BuildingLayer,
)
import shapely.geometry as geo
import shapely
import numpy as np
import copy
import math
from rtree import index
from scipy.spatial.transform import Rotation
import yaml
from functools import partial
import cv2
import parse

SEMANTICS_TO_COLOR = np.array(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
)
LABEL_TO_COLOR = {
    "frontier": [1, 0, 0],
    "place": [0, 0, 0],
    "anti_frontier": [1, 1, 1],
    "predicted_place": [1, 0, 1],
}

TRAVERSABLE_SEMANTICS = ["Unknown"]
MOVEABLE_OBJECTS = ["mine"]
SURFACE_OBJECTS = ["table"]

ROOM_ID_TO_LABEL = {
    0: "office",
    1: "hallway",
    2: "conference_room",
    3: "bathroom",
    4: "kitchen",
    5: "stairs",
    6: "bedroom",
    99: "unknown",
}

ROOM_LABEL_TO_ID = {
    "office": 0,
    "hallway": 1,
    "conference_room": 2,
    "bathroom": 3,
    "unknown": 99,
}


# ade20k_mit labels
SEMANTIC_ID_TO_LABEL = {
    0: "unknown",
    1: "sky",
    2: "tree",
    3: "water",
    4: "ground",
    5: "grass",
    6: "sand",
    7: "sidewalk",
    8: "dock",
    9: "road",
    10: "path",
    11: "vehicle",
    12: "building",
    13: "shelter",
    14: "signal",
    15: "rock",
    16: "fence",
    17: "boat",
    18: "sign",
    19: "hill",
    20: "bridge",
    21: "wall",
    22: "floor",
    23: "ceiling",
    24: "door",
    25: "stairs",
    26: "pole",
    27: "rail",
    28: "structure",
    29: "window",
    30: "surface",
    31: "flora",
    32: "flower",
    33: "bed",
    34: "box",
    35: "storage",
    36: "barrel",
    37: "bag",
    38: "basket",
    39: "seating",
    40: "flag",
    41: "decor",
    42: "light",
    43: "appliance",
    44: "trash",
    45: "bicycle",
    46: "food",
    47: "clothes",
    48: "thing",
    49: "animal",
    50: "human",
}


def center_w_h_to_bb(center, w, h):
    return np.array(
        [
            [center[0] - w, center[1] - h],
            [center[0] + w, center[1] - h],
            [center[0] + w, center[1] + h],
            [center[0] - w, center[1] + h],
        ]
    )


def str_to_ns_value(s):
    p = parse.parse("{}({})", s)
    key = p.fixed[0]
    idx = int(p.fixed[1])
    ns = spark_dsg.NodeSymbol(key, idx)
    return ns.value


def load_semantic_id_map_from_label_space(fn):
    with open(fn, "r") as fo:
        labelspace_yaml = yaml.safe_load(fo)
    id_to_label = {e["label"]: e["name"] for e in labelspace_yaml["label_names"]}
    return id_to_label


def load_inverse_semantic_id_map_from_label_space(fn):
    with open(fn, "r") as fo:
        labelspace_yaml = yaml.safe_load(fo)
    label_to_id = {e["name"]: e["label"] for e in labelspace_yaml["label_names"]}
    return label_to_id


def quaternion_to_yaw(quaternion):
    # Ensure the quaternion is normalized
    quaternion /= np.linalg.norm(quaternion)

    # Extract components
    w, x, y, z = quaternion

    # Calculate yaw angle in radians
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    return yaw


def rotate_2d_bounding_box(bbox3d, yaw_radians):
    # Bounding box format: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    bbox_new = bbox3d.copy()
    bbox = bbox3d[:, :2]
    # Calculate the center of the bounding box
    center_x = (bbox[0][0] + bbox[2][0]) / 2.0
    center_y = (bbox[0][1] + bbox[2][1]) / 2.0

    # Translate the bounding box to the origin
    translated_bbox = [(x - center_x, y - center_y) for x, y in bbox]

    # Rotate each corner of the bounding box by the yaw angle
    rotated_bbox = [
        (
            x * math.cos(yaw_radians) - y * math.sin(yaw_radians),
            x * math.sin(yaw_radians) + y * math.cos(yaw_radians),
        )
        for x, y in translated_bbox
    ]

    # Translate the rotated bounding box back to its original position
    rotated_bbox = [(x + center_x, y + center_y) for x, y in rotated_bbox]

    bbox_new[:, :2] = rotated_bbox

    return bbox_new


def attrs_to_bb(attrs):
    # mi = attrs.bounding_box.min
    # ma = attrs.bounding_box.max
    # bb = np.array(
    #    [
    #        [ma[0], ma[1], mi[2]],
    #        [ma[0], mi[1], mi[2]],
    #        [mi[0], mi[1], mi[2]],
    #        [mi[0], ma[1], mi[2]],
    #        [ma[0], ma[1], ma[2]],
    #        [ma[0], mi[1], ma[2]],
    #        [mi[0], ma[1], ma[2]],
    #        [mi[0], mi[1], ma[2]],
    #    ]
    # )
    bb = np.array(attrs.bounding_box.corners())
    return bb


def recompute_place_connectivity(sg, buffer=0.5):
    if sg.places_2d.boundaries_shapely is None:
        sg.make_shapely_boundaries()
    sg = copy.copy(sg)
    sg.places_2d = copy.copy(sg.places_2d)
    sg.places_2d.connectivity = []
    sg.places_2d.neighbors = {}
    for i in range(len(sg.places_2d.indices)):
        sg.places_2d.neighbors[i] = []

    place_index = index.Index()

    # Insert polygons into the indices
    bounds = [bb.buffer(buffer) for bb in sg.places_2d.boundaries_shapely]
    for idx, place in enumerate(bounds):
        if any(np.isnan(place.bounds)):
            continue
        place_index.insert(idx, place.bounds)

    for i in range(len(sg.places_2d.indices)):
        poly_i = sg.places_2d.boundaries_shapely[i]
        poly_i_buf = sg.places_2d.boundaries_shapely[i].buffer(buffer)
        bounds_i = poly_i.bounds
        if any(np.isnan(bounds_i)):
            continue

        possible_neighbors = place_index.intersection(bounds_i)
        for j in possible_neighbors:
            if i == j:
                continue
            poly_j = sg.places_2d.boundaries_shapely[j]
            if poly_i_buf.intersects(poly_j):
                # add connection
                sg.places_2d.connectivity.append((i, j))
                sg.places_2d.neighbors[i].append(j)

    return sg


def remove_neighbor_edge(pdsg, s, t):
    """Return a new dsg without edge s --> t"""

    pdsg = copy.copy(pdsg)
    pdsg.places_2d = copy.copy(pdsg)
    ns = pdsg.places_2d.neighbors
    pdsg.places_2d.neighbors = copy.copy(pdsg.places_2d.ns)

    pdsg.places_2d.neighbors[s] = [ix for ix in ns[s] if ix != t]
    return pdsg


def remove_place_edge(sg, i, j):

    sg = copy.copy(sg)
    sg.places_2d = copy.copy(sg.places_2d)
    sg.places_2d.connectivity = [
        e for e in sg.places_2d.connectivity if e[0] not in [i, j] or e[1] not in [i, j]
    ]
    sg.places_2d.neighbors = copy.copy(sg.places_2d.neighbors)
    sg.places_2d.neighbors[i] = [n for n in sg.places_2d.neighbors[i] if n != j]
    sg.places_2d.neighbors[j] = [n for n in sg.places_2d.neighbors[j] if n != i]
    return sg


def update_connectivity_by_traversable(pdsg):
    """Remove edges that have at least one endpoint marked untraversable"""

    sg = copy.copy(pdsg)
    sg.places_2d.connectivity = copy.copy(sg.places_2d.connectivity)
    sg.places_2d.neighbors = copy.copy(sg.places_2d.neighbors)

    sg.places_2d.connectivity = [
        e
        for e in sg.places_2d.connectivity
        if sg.places_2d.traversable[e[0]] and sg.places_2d.traversable[e[1]]
    ]
    sg.places_2d.neighbors = {
        i: set(
            filter(
                lambda j: sg.places_2d.traversable[j] and sg.places_2d.traversable[i],
                ni,
            )
        )
        for i, ni in sg.places_2d.neighbors.items()
    }

    return sg


def update_traversability(
    pdsg: PyDSG,
    place_ixs,
    obstruction_polys,
    obstruction_buffer,
    place_min_area_ratio,
    height_threshold,
    obstruction_heights,
):
    pdsg = copy.copy(pdsg)
    pdsg.places_2d.traversable = pdsg.places_2d.traversable.copy()
    pdsg.places_2d.boundary_shapely = pdsg.places_2d.boundary_shapely.copy()

    obstacle_index = index.Index()
    valid_region_index = index.Index()

    # Insert polygons into the indices
    bad_bounds = [bb.buffer(obstruction_buffer) for bb in obstruction_polys]
    for idx, obstacle in enumerate(bad_bounds):
        obstacle_index.insert(idx, obstacle.bounds)

    for idx in place_ixs:
        good_boundary = pdsg.places_2d.boundary_shapely[idx]
        if good_boundary.is_empty:
            print("Empty boundary")
        else:
            valid_region_index.insert(idx, good_boundary.bounds)

    for ix in place_ixs:
        z = pdsg.places_2d.center[ix, 2]
        good_boundary = pdsg.places_2d.boundary_shapely[ix]
        if good_boundary.is_empty:
            print("Empty boundary")
        else:
            potential_obstacle_indices = list(
                obstacle_index.intersection(good_boundary.bounds)
            )

            for ob_idx in potential_obstacle_indices:
                bad_boundary = bad_bounds[ob_idx]
                obs_z = obstruction_heights[ob_idx]
                place_conflict_tolerance = 1  # If there is a non-traversable object *below* us, at what distance do we consider it blocking us
                if z + height_threshold > obs_z > z - place_conflict_tolerance:
                    try:
                      good_boundary = good_boundary - bad_boundary
                    except Exception as ex:
                      print(ex)
                      print('skipping')
                    # if good_boundary.intersects(bad_boundary): # TODO: figure out why this was crashing so we can re-add it?
                    #     good_boundary = good_boundary - bad_boundary
        pdsg.places_2d.boundary_shapely[ix] = good_boundary
        if good_boundary.area / pdsg.places_2d.original_area[ix] < place_min_area_ratio:
            pdsg.places_2d.traversable[ix] = False
        else:
            pdsg.places_2d.traversable[ix] = True

    return pdsg


def update_semantic_place_traversability_idx(
    pdsg,
    traversable_ixs_for_update,
    untraversable_ixs_for_update,
    place_buffer,
    place_min_area_ratio,
    place_height_threshold=2,
):
    """Return new pdsg where the places in traversable_ixs_for_update have been eroded by the buffered boundaries of untraversable_ixs_for_update"""

    untraversable_polys = [
        pdsg.places_2d.boundary_shapely[ix] for ix in untraversable_ixs_for_update
    ]
    untraversable_poly_heights = [
        pdsg.places_2d.center[ix, 2] for ix in untraversable_ixs_for_update
    ]
    return update_traversability(
        pdsg,
        traversable_ixs_for_update,
        untraversable_polys,
        place_buffer,
        place_min_area_ratio,
        place_height_threshold,
        untraversable_poly_heights,
    )


def update_object_traversability_idx(
    pdsg,
    place_ixs_for_update,
    object_ixs_for_update,
    obstruction_buffer,
    place_min_area_ratio,
    object_height_threshold=2,
):
    """Return new pdsg where the places in place_ixs_for_update have been eroded by the buffered object boundaries of object_ixs_for_update"""

    untraversable_polys = [
        pdsg.objects.boundary_shapely[ix] for ix in object_ixs_for_update
    ]
    untraversable_poly_heights = [
        pdsg.objects.center[ix, 2] for ix in object_ixs_for_update
    ]
    return update_traversability(
        pdsg,
        place_ixs_for_update,
        untraversable_polys,
        obstruction_buffer,
        place_min_area_ratio,
        object_height_threshold,
        untraversable_poly_heights,
    )


# def add_traversability_info(pdsg, traversable_semantics, moveable_objects):
def get_traversable_dsg(
    pdsg: PyDSG,
    traversable_semantics,
    moveable_objects,
    place_buffer=0.1,
    object_buffer=0.1,
    place_min_area_ratio=1.0,
):
    # place_buffer -- amount to buffer untraversable places when checking whether places are traversable
    # object_buffer -- amount to buffer immovable objects when checking whether places are traversable
    # place_min_area_ratio -- What ratio of obstacle coverage makes a place count as "blocked".
    # 0 means never blocked, 1 means any object blocks

    traversability = np.ones(len(pdsg.places_2d.hydra_symbol), dtype=bool)

    pdsg = copy.copy(pdsg)
    pdsg.places_2d.traversable = traversability

    # Update semantics for places that are inherently untraversable
    good_place_ixs = []
    bad_place_ixs = []
    for ix in range(len(pdsg.places_2d)):
        if pdsg.places_2d.semantic_label[ix] not in traversable_semantics:
            pdsg.places_2d.traversable[ix] = False
            bad_place_ixs.append(ix)
        else:
            good_place_ixs.append(ix)

    # Erode traversable places by buffered untraversable places
    # profile indicates this is slow
    place_traversable_dsg = update_semantic_place_traversability_idx(
        pdsg, good_place_ixs, bad_place_ixs, place_buffer, place_min_area_ratio
    )

    # Identify which objects are not movable, and therefore should always be eroded from the passable places
    object_ixs_for_update = []
    for ix, label in enumerate(pdsg.objects.semantic_label):
        if label not in moveable_objects:
            object_ixs_for_update.append(ix)

    # Erode traversable places by buffered immpovable objects
    traversable_dsg = update_object_traversability_idx(
        place_traversable_dsg,
        good_place_ixs,
        object_ixs_for_update,
        object_buffer,
        place_min_area_ratio,
    )

    return traversable_dsg


def should_include_place(n):
    return len(n.attributes.boundary) > 2


def make_shapely_boundary(b):
    if len(b) == 1:
        return geo.Point(b).buffer(0.1)
    elif len(b) == 2:
        return geo.LineString(b).buffer(0.1)
    else:
        return geo.Polygon(b)


def make_place_3d(
    G,
    semantic_id_to_label,
    semantics_to_color,
    pid_to_index,
    p,
):
    if str(p.id) not in pid_to_index.keys():
        print("Didn't find place 3d!")
        return

    attrs = p.attributes
    boundary = np.array(geo.Point(attrs.position).buffer(attrs.distance).exterior.xy).T

    if p.attributes.real_place:
        label = "place"
    elif p.attributes.is_predicted:
        label = "predicted_place"
    elif p.attributes.anti_frontier:
        label = "anti_frontier"
    else:
        label = "frontier"
    color = semantics_to_color[label]

    is_frontier = (
        (not attrs.real_place)
        and (not attrs.is_predicted)
        and (not attrs.anti_frontier)
    )

    place = Place3d(
        hydra_symbol=str(p.id),
        from_hydra=True,
        center=attrs.position,
        boundary=boundary,
        radius=attrs.distance,
        semantic_label=label,
        semantic_color=color,
        predicted_place=attrs.is_predicted,
        frontier=is_frontier,
    )
    return place


def make_place_2d(G, semantic_id_to_label, semantics_to_color, pid_to_index, p):
    if str(p.id) not in pid_to_index.keys():
        print("Didn't find place 3d!")
        return

    attrs = p.attributes
    boundary = np.array(p.attributes.boundary)[:, :2]
    boundary_shapely = make_shapely_boundary(boundary)
    shapely.prepare(boundary_shapely)
    area = boundary_shapely.area

    label = semantic_id_to_label[p.attributes.semantic_label]
    color = SEMANTICS_TO_COLOR[p.attributes.semantic_label % len(SEMANTICS_TO_COLOR)]

    place = Place2d(
        hydra_symbol=str(p.id),
        from_hydra=True,
        center=attrs.position,
        boundary=boundary,
        boundary_shapely=boundary_shapely,
        original_area=area,
        semantic_label=label,
        semantic_color=color,
    )
    return place


def make_object(
    G,
    semantic_id_to_label,
    semantics_to_color,
    pid_to_index,
    oid_to_index,
    moveable_objects,
    surface_objects,
    o,
):
    if np.isnan(np.array(o.attributes.bounding_box.corners())).any():
        print("Found nan in object bounding box. Skipping!")
        return

    attrs = o.attributes
    label = semantic_id_to_label[attrs.semantic_label]
    color = o.attributes.color

    rotation = o.attributes.world_R_object
    bb_rotation = o.attributes.bounding_box.world_R_center
    r = Rotation.from_matrix(bb_rotation)
    bb_yaw_radians = r.as_euler("zyx")[0]
    rotation = [rotation.w, rotation.x, rotation.y, rotation.z]

    # bb = o.attributes.position + attrs_to_bb(o.attributes)
    bb = attrs_to_bb(o.attributes)
    bb = rotate_2d_bounding_box(bb, bb_yaw_radians)

    obj = Object(
        hydra_symbol=str(o.id),
        from_hydra=True,
        center=attrs.position,
        semantic_label=label,
        semantic_color=color / 255,
        orientation=rotation,
        movable=label in moveable_objects,
        surface=label in surface_objects,
        box=bb,
        boundary_shapely=geo.Polygon(bb[:4, :4]),
    )

    return obj


def make_room(G, room_id_to_label, r):

    attrs = r.attributes
    label = room_id_to_label[attrs.semantic_label]
    color = r.attributes.color

    place_points = np.array(
        [G.get_node(c).attributes.position[:2] for c in r.children()]
    )
    if len(place_points) == 0:
        bb = center_w_h_to_bb(attrs.position[:2], 1, 1)
    else:
        bb = cv2.boxPoints(cv2.minAreaRect(place_points.astype(np.float32)))

    label_confidences = r.attributes.semantic_class_probabilities

    room = Room(
        hydra_symbol=str(r.id),
        from_hydra=True,
        center=attrs.position,
        semantic_label=label,
        semantic_color=color / 255,
        box=bb,
        boundary_shapely=geo.Polygon(bb),
        label_confidences=label_confidences,
        predicted=attrs.is_predicted,
    )

    return room


def make_building(G, b):
    attrs = b.attributes
    label = attrs.semantic_label
    color = np.array((0, 0, 0))
    building = Building(
        hydra_symbol=str(b.id),
        from_hydra=True,
        center=attrs.position,
        semantic_label=label,
        semantic_color=color / 255,
    )
    return building


def get_parents(G, index_to_hydra_symbol, layer_name):
    return {
        str(n.id): [index_to_hydra_symbol[n.get_parent()]] if n.has_parent() else []
        for n in G.get_layer(layer_name).nodes
    }


def get_siblings(G, index_to_hydra_symbol, layer_name):
    return {
        str(n.id): [
            index_to_hydra_symbol[m] for m in n.siblings() if m in index_to_hydra_symbol
        ]
        for n in G.get_layer(layer_name).nodes
    }


def get_children(G, index_to_hydra_symbol, layer_name):
    return {
        str(n.id): [
            index_to_hydra_symbol[m] for m in n.children() if m in index_to_hydra_symbol
        ]
        for n in G.get_layer(layer_name).nodes
    }


def get_sibling_probability(G, index_to_hydra_symbol, layer_name, symbols):
    probability_map = {}
    for ix in range(len(symbols)):
        s1 = symbols[ix]
        for jx in range(ix + 1, len(symbols)):
            s2 = symbols[jx]
            edge = G.find_edge(str_to_ns_value(s1), str_to_ns_value(s2))
            if edge is None:
                continue
            if edge.info.weighted:
                probability_map[(s1, s2)] = 1
                probability_map[(s2, s1)] = 1
            else:
                probability_map[(s1, s2)] = edge.info.weight
                probability_map[(s2, s1)] = edge.info.weight


def spark_dsg_to_pydsg(
    G,
    traversable_semantics=None,
    semantic_id_to_label=None,
    moveable_objects=None,
    surface_objects=None,
    place_min_area_ratio=0.0,
    room_id_to_label=None,
):
    if traversable_semantics is None:
        traversable_semantics = TRAVERSABLE_SEMANTICS
    if semantic_id_to_label is None:
        semantic_id_to_label = SEMANTIC_ID_TO_LABEL
    if moveable_objects is None:
        moveable_objects = MOVEABLE_OBJECTS
    if surface_objects is None:
        surface_objects = SURFACE_OBJECTS
    if room_id_to_label is None:
        room_id_to_label = ROOM_ID_TO_LABEL

    index_to_pid = {}
    pid_to_index = {}
    ix = 0
    for n in G.get_layer(spark_dsg.DsgLayers.PLACES).nodes:
        pid_to_index[str(n.id)] = ix
        index_to_pid[ix] = str(n.id)
        ix += 1

    index_to_pid_2d = {}
    pid_to_index_2d = {}
    ix = 0
    for n in G.get_layer(spark_dsg.DsgLayers.MESH_PLACES).nodes:
        if should_include_place(n):
            pid_to_index_2d[str(n.id)] = ix
            index_to_pid_2d[ix] = str(n.id)
            ix += 1

    oid_to_index = {}
    index_to_oid = {}
    ix = 0
    for n in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        oid_to_index[str(n.id)] = ix
        index_to_oid[ix] = str(n.id)
        ix += 1

    rid_to_index = {}
    index_to_rid = {}
    ix = 0
    for n in G.get_layer(spark_dsg.DsgLayers.ROOMS).nodes:
        rid_to_index[str(n.id)] = ix
        index_to_rid[ix] = str(n.id)
        ix += 1

    bid_to_index = {}
    index_to_bid = {}
    ix = 0
    for n in G.get_layer(spark_dsg.DsgLayers.BUILDINGS).nodes:
        bid_to_index[str(n.id)] = ix
        index_to_bid[ix] = str(n.id)
        ix += 1

    gtsam_symbol_to_index = {}
    index_to_hydra_symbol = {}
    for n in G.get_layer(spark_dsg.DsgLayers.PLACES).nodes:
        gtsam_symbol_to_index[str(n.id)] = n.id.value
        index_to_hydra_symbol[n.id.value] = str(n.id)
    for n in G.get_layer(spark_dsg.DsgLayers.MESH_PLACES).nodes:
        gtsam_symbol_to_index[str(n.id)] = n.id.value
        index_to_hydra_symbol[n.id.value] = str(n.id)
    for n in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        gtsam_symbol_to_index[str(n.id)] = n.id.value
        index_to_hydra_symbol[n.id.value] = str(n.id)
    for n in G.get_layer(spark_dsg.DsgLayers.ROOMS).nodes:
        gtsam_symbol_to_index[str(n.id)] = n.id.value
        index_to_hydra_symbol[n.id.value] = str(n.id)
    for n in G.get_layer(spark_dsg.DsgLayers.BUILDINGS).nodes:
        gtsam_symbol_to_index[str(n.id)] = n.id.value
        index_to_hydra_symbol[n.id.value] = str(n.id)

    build_place_3d = partial(
        make_place_3d,
        G,
        semantic_id_to_label,
        LABEL_TO_COLOR,
        pid_to_index,
    )
    places_3d = list(map(build_place_3d, G.get_layer(spark_dsg.DsgLayers.PLACES).nodes))
    place_layer_3d = to_soa(places_3d, Place3dLayer)
    place_layer_3d.hydra_index_to_local_index = pid_to_index
    place_layer_3d.local_index_to_hydra_index = index_to_pid

    get_parents_p = partial(get_parents, G, index_to_hydra_symbol)
    get_siblings_p = partial(get_siblings, G, index_to_hydra_symbol)
    get_children_p = partial(get_children, G, index_to_hydra_symbol)

    place_layer_3d.parent_dict = get_parents_p(spark_dsg.DsgLayers.PLACES)
    place_layer_3d.sibling_dict = get_siblings_p(spark_dsg.DsgLayers.PLACES)
    place_layer_3d.children_dict = get_children_p(spark_dsg.DsgLayers.PLACES)

    build_place_2d = partial(
        make_place_2d, G, semantic_id_to_label, LABEL_TO_COLOR, pid_to_index_2d
    )
    places_2d = list(
        map(build_place_2d, G.get_layer(spark_dsg.DsgLayers.MESH_PLACES).nodes)
    )
    place_layer_2d = to_soa(places_2d, Place2dLayer)
    place_layer_2d.hydra_index_to_local_index = pid_to_index_2d
    place_layer_2d.local_index_to_hydra_index = index_to_pid_2d

    place_layer_2d.parent_dict = get_parents_p(spark_dsg.DsgLayers.MESH_PLACES)
    place_layer_2d.sibling_dict = get_siblings_p(spark_dsg.DsgLayers.MESH_PLACES)
    place_layer_2d.children_dict = get_children_p(spark_dsg.DsgLayers.MESH_PLACES)

    build_object = partial(
        make_object,
        G,
        semantic_id_to_label,
        SEMANTICS_TO_COLOR,
        pid_to_index,
        oid_to_index,
        moveable_objects,
        surface_objects,
    )
    objects = list(map(build_object, G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes))
    object_layer = to_soa(objects, ObjectLayer)
    object_layer.hydra_index_to_local_index = oid_to_index
    object_layer.local_index_to_hydra_index = index_to_oid

    object_layer.parent_dict = get_parents_p(spark_dsg.DsgLayers.OBJECTS)
    object_layer.sibling_dict = get_siblings_p(spark_dsg.DsgLayers.OBJECTS)
    object_layer.children_dict = get_children_p(spark_dsg.DsgLayers.OBJECTS)

    build_room = partial(make_room, G, room_id_to_label)
    rooms = list(map(build_room, G.get_layer(spark_dsg.DsgLayers.ROOMS).nodes))
    room_layer = to_soa(rooms, RoomLayer)
    room_layer.hydra_index_to_local_index = rid_to_index
    room_layer.local_index_to_hydra_index = index_to_rid

    room_layer.parent_dict = get_parents_p(spark_dsg.DsgLayers.ROOMS)
    room_layer.sibling_dict = get_siblings_p(spark_dsg.DsgLayers.ROOMS)
    room_layer.children_dict = get_children_p(spark_dsg.DsgLayers.ROOMS)
    room_layer.edge_probability_dict = get_sibling_probability(
        G, room_id_to_label, spark_dsg.DsgLayers.ROOMS, room_layer.hydra_symbol
    )

    n_rooms = len(rooms)
    room_edge_mat = np.eye(n_rooms)
    for si in room_layer.sibling_dict.keys():
        for sj in room_layer.sibling_dict[si]:
            room_edge_mat[
                room_layer.hydra_index_to_local_index[si],
                room_layer.hydra_index_to_local_index[sj],
            ] = 1
    room_layer.edge_probabilities = room_edge_mat

    build_building = partial(make_building, G)
    buildings = list(
        map(build_building, G.get_layer(spark_dsg.DsgLayers.BUILDINGS).nodes)
    )
    building_layer = to_soa(buildings, BuildingLayer)
    building_layer.hydra_index_to_local_index = bid_to_index
    building_layer.local_index_to_hydra_index = index_to_bid

    building_layer.parent_dict = get_parents_p(spark_dsg.DsgLayers.BUILDINGS)
    building_layer.sibling_dict = get_siblings_p(spark_dsg.DsgLayers.BUILDINGS)
    building_layer.children_dict = get_children_p(spark_dsg.DsgLayers.BUILDINGS)

    pdsg = PyDSG(
        objects=object_layer,
        places_2d=place_layer_2d,
        places_3d=place_layer_3d,
        rooms=room_layer,
        buildings=building_layer,
    )

    pdsg.gtsam_symbol_to_index = gtsam_symbol_to_index
    pdsg.index_to_hydra_symbol = index_to_hydra_symbol

    pdsg = get_traversable_dsg(
        pdsg,
        traversable_semantics,
        moveable_objects,
        place_min_area_ratio=place_min_area_ratio,
    )

    return pdsg


def add_edges_from_pydsg(G, layer, sibling_probabilities=None):

    print("edges: ", layer.sibling_dict)
    if layer.sibling_dict is not None:
        for pi in layer.hydra_symbol:
            if pi not in layer.sibling_dict:
                continue
            for pj in layer.sibling_dict[pi]:
                if sibling_probabilities is None:
                    G.insert_edge(str_to_ns_value(pi), str_to_ns_value(pj))
                else:
                    print("edge probabilities:", sibling_probabilities)
                    ea = spark_dsg.EdgeAttributes()
                    ea.weight = sibling_probabilities[(pi, pj)]
                    G.insert_edge(str_to_ns_value(pi), str_to_ns_value(pj), ea)

    if layer.parent_dict is not None:
        for pi in layer.hydra_symbol:
            if pi not in layer.parent_dict:
                continue
            for pj in layer.parent_dict[pi]:
                G.insert_edge(str_to_ns_value(pi), str_to_ns_value(pj))

    if layer.children_dict is not None:
        for pi in layer.hydra_symbol:
            if pi not in layer.children_dict:
                continue
            for pj in layer.children_dict[pi]:
                G.insert_edge(str_to_ns_value(pi), str_to_ns_value(pj))


def py_to_spark_place2d(p, label_to_semantic_id):
    attrs = spark_dsg.Place2dNodeAttributes()
    attrs.position = p.center
    attrs.name = p.hydra_symbol
    boundary = np.zeros((len(p.boundary), 3))
    boundary[:, :2] = p.boundary
    boundary[:, 2] = p.center[2]
    attrs.boundary = boundary
    attrs.semantic_label = label_to_semantic_id[p.semantic_label]
    attrs.color = p.semantic_color * 255
    return attrs


def py_to_spark_place3d(p, label_to_semantic_id):
    attrs = spark_dsg.PlaceNodeAttributes()
    attrs.name = p.hydra_symbol
    attrs.position = p.center
    # attrs.semantic_label = label_to_semantic_id[p.semantic_label]
    attrs.semantic_label = 0 if p.semantic_label == "place" else 1
    attrs.color = p.semantic_color * 255
    attrs.distance = p.radius
    # attrs.real_place = True if p.semantic_label == "place" else False
    attrs.real_place = (
        not p.predicted_place
        and not p.frontier
        and not p.semantic_label == "anti_frontier"
    )
    attrs.predicted_place = p.predicted_place
    attrs.anti_frontier = p.semantic_label == "anti_frontier"
    return attrs


def py_to_spark_objects(o, label_to_semantic_id):
    attrs = spark_dsg.ObjectNodeAttributes()
    attrs.name = o.hydra_symbol
    attrs.position = o.center
    attrs.semantic_label = label_to_semantic_id[o.semantic_label]
    attrs.color = o.semantic_color * 255
    attrs.bounding_box.min = np.min(o.box, axis=0)
    attrs.bounding_box.max = np.max(o.box, axis=0)
    return attrs


def py_to_spark_rooms(r, room_label_to_id):

    attrs = spark_dsg.RoomNodeAttributes()
    attrs.name = r.hydra_symbol if r.semantic_label != "unknown" else "?"
    print("\n\n\nAttr name: ", attrs.name)
    attrs.position = r.center

    bb_center = np.mean(r.box, axis=0)
    bb_width = np.max(r.box, axis=0) - np.min(r.box, axis=0)

    attrs.bounding_box = spark_dsg.BoundingBox(bb_width, bb_center)
    attrs.color = r.semantic_color * 255
    attrs.semantic_label = room_label_to_id[r.semantic_label]
    attrs.semantic_class_probabilities = r.label_confidences
    attrs.is_predicted = r.predicted
    return attrs


def pydsg_to_spark_dsg(
    pdsg, label_to_semantic_id, room_label_to_id, G=None, add_filter=lambda x: True
):

    if G is None:
        layers = [
            spark_dsg.DsgLayers.PLACES,
            spark_dsg.DsgLayers.MESH_PLACES,
            spark_dsg.DsgLayers.OBJECTS,
            spark_dsg.DsgLayers.ROOMS,
            spark_dsg.DsgLayers.BUILDINGS,
        ]
        G = spark_dsg.DynamicSceneGraph(layers)

    if pdsg.places_2d is not None:
        for p in pdsg.places_2d:
            if not add_filter(p):
                continue
            attrs = py_to_spark_place2d(p, label_to_semantic_id)
            G.add_node(
                spark_dsg.DsgLayers.MESH_PLACES, str_to_ns_value(p.hydra_symbol), attrs
            )
        add_edges_from_pydsg(G, pdsg.places_2d)

    if pdsg.places_3d is not None:
        for p in pdsg.places_3d:
            if not add_filter(p):
                continue
            attrs = py_to_spark_place3d(p, label_to_semantic_id)
            G.add_node(
                spark_dsg.DsgLayers.PLACES, str_to_ns_value(p.hydra_symbol), attrs
            )
        add_edges_from_pydsg(G, pdsg.places_3d)

    if pdsg.objects is not None:
        for o in pdsg.objects:
            if not add_filter(o):
                continue
            attrs = py_to_spark_objects(o, label_to_semantic_id)
            G.add_node(
                spark_dsg.DsgLayers.OBJECTS, str_to_ns_value(o.hydra_symbol), attrs
            )
        add_edges_from_pydsg(G, pdsg.objects)

    if pdsg.rooms is not None:
        for r in pdsg.rooms:
            if not add_filter(r):
                continue
            attrs = py_to_spark_rooms(r, room_label_to_id)
            G.add_node(
                spark_dsg.DsgLayers.ROOMS, str_to_ns_value(r.hydra_symbol), attrs
            )
        add_edges_from_pydsg(
            G, pdsg.rooms, sibling_probabilities=pdsg.rooms.edge_probability_dict
        )

    # TODO: buildings
    return G


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    labelspace_path = "/home/ubuntu/catkin_ws/src/hydra/config/label_spaces/ade20k_full_label_space.yaml"
    traversable_semantics = [
        "floor",
        "road",
        "sidewalk",
        "field",
        "path",
        "runway",
        "dirt",
        "land",
    ]
    semantic_id_to_label = load_semantic_id_map_from_label_space(labelspace_path)

    test_dsg_fn = "/home/ubuntu/catkin_ws/src/active_dsg_launch/dsg.json"
    G = dsg.DynamicSceneGraph.load(test_dsg_fn)

    pdsg = spark_dsg_to_pydsg(
        G,
        semantic_id_to_label=semantic_id_to_label,
        traversable_semantics=traversable_semantics,
    )

    from dsg_plotter import plot_dsg_places, plot_dsg_objects, plot_dsg_rooms

    plt.ion()
    plot_dsg_places(pdsg, with_edges=True)
    plot_dsg_objects(pdsg, plot_indices=True)
    plot_dsg_rooms(pdsg)
    plt.show()
