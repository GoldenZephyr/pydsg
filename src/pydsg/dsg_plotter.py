import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geo
from matplotlib.collections import LineCollection


def plot_nx(graph, color_map, position_map):
    colors = []
    positions = []

    for n in graph.nodes:
        colors.append(color_map[n])
        positions.append(position_map[n])

    positions = np.array(positions)
    plt.scatter(positions[:, 0], positions[:, 1], c=colors)

    edges = []
    for e in graph.edges:
        p0 = position_map[e[0]]
        p1 = position_map[e[1]]
        edges.append((p0, p1))

    lc = LineCollection(edges, color="k")
    plt.gca().add_collection(lc)


def plot_dsg_predicted_places(
    dsg,
    with_edges=True,
    plot_mask=None,
    plot_indices=False,
    region=None,
    boundary_lw=1,
    edge_lw=1,
    with_centers=True,
    color_center_by_semantics=False,
):

    if dsg.places_3d is None or dsg.places_3d.center is None:
        return
    if plot_mask is None:
        plot_mask = np.ones(len(dsg.places_3d)).astype(bool)

    plot_mask_symbol = {s: pm for s, pm in zip(dsg.places_3d.hydra_symbol, plot_mask)}

    if region is not None:
        distances = np.linalg.norm(dsg.places_3d.center[:, :2] - region[0], axis=1)
        close_enough = distances < region[1]
        plot_mask = np.logical_and(plot_mask, close_enough)

    rc = np.array(dsg.places_3d.center)
    rc_clip = rc[plot_mask]
    if with_centers:
        if color_center_by_semantics:
            plt.scatter(
                rc_clip[:, 0],
                rc_clip[:, 1],
                c=np.array(dsg.places_3d.semantic_color)[plot_mask],
            )
        else:
            plt.scatter(rc_clip[:, 0], rc_clip[:, 1], color="k")

    if with_edges:
        lines = []
        neighbors = dsg.places_3d.sibling_dict
        for node_s in neighbors:
            if node_s not in plot_mask_symbol:
                print("WARNING: node in sibling but not node list")
                continue
            if not plot_mask_symbol[node_s]:
                continue
            start = dsg.places_3d[node_s].center[:2]
            for node_t in neighbors[node_s]:
                end = dsg.places_3d[node_t].center[:2]

                if node_t not in plot_mask_symbol:
                    print("WARNING: node in sibling but not node list")
                    continue
                if plot_mask_symbol[node_t]:
                    lines.append((start, end))

        lc = LineCollection(lines, linewidth=edge_lw, color="k")
        plt.gca().add_collection(lc)

    if plot_indices:
        for ind, center in zip(
            np.array(dsg.places_3d.hydra_symbol)[plot_mask],
            dsg.places_3d.center[plot_mask],
        ):
            plt.text(center[0], center[1], f"{ind}")


def plot_dsg_places(
    dsg,
    with_boundary=True,
    with_edges=True,
    plot_mask=None,
    plot_indices=False,
    region=None,
    boundary_lw=1,
    edge_lw=1,
    with_centers=True,
    color_center_by_semantics=False,
):

    if dsg.places_2d is None or dsg.places_2d.center is None:
        return
    if plot_mask is None:
        plot_mask = np.ones(len(dsg.places_2d)).astype(bool)

    plot_mask_symbol = {s: pm for s, pm in zip(dsg.places_2d.hydra_symbol, plot_mask)}

    if region is not None:
        distances = np.linalg.norm(dsg.places_2d.center[:, :2] - region[0], axis=1)
        close_enough = distances < region[1]
        plot_mask = np.logical_and(plot_mask, close_enough)

    if with_boundary:
        for ix, r, c in zip(
            range(len(dsg.places_2d)),
            dsg.places_2d.boundary_shapely,
            dsg.places_2d.semantic_color,
        ):
            if plot_mask[ix]:
                if isinstance(r, geo.Polygon):
                    x, y = r.exterior.xy
                    plt.plot(x, y, color=c, linewidth=boundary_lw)
                else:
                    for g in r.geoms:
                        x, y = g.exterior.xy
                        plt.plot(x, y, color=c, linewidth=boundary_lw)

    rc = np.array(dsg.places_2d.center)
    rc_clip = rc[plot_mask]
    if with_centers:
        if color_center_by_semantics:
            plt.scatter(
                rc_clip[:, 0],
                rc_clip[:, 1],
                c=np.array(dsg.places_2d.semantic_color)[plot_mask],
            )
        else:
            plt.scatter(rc_clip[:, 0], rc_clip[:, 1], color="k")

    if with_edges:
        lines = []
        neighbors = dsg.places_2d.sibling_dict
        for node_s in neighbors:
            if node_s not in plot_mask_symbol:
                print("WARNING: node in sibling but not node list")
                continue
            if not plot_mask_symbol[node_s]:
                continue
            start = dsg.places_2d[node_s].center[:2]
            for node_t in neighbors[node_s]:
                end = dsg.places_2d[node_t].center[:2]

                if node_t not in plot_mask_symbol:
                    print("WARNING: node in sibling but not node list")
                    continue
                if plot_mask_symbol[node_t]:
                    lines.append((start, end))

        lc = LineCollection(lines, linewidth=edge_lw, color="k")
        plt.gca().add_collection(lc)

    if plot_indices:
        for ind, center in zip(
            np.array(dsg.places_2d.hydra_symbol)[plot_mask],
            dsg.places_2d.center[plot_mask],
        ):
            plt.text(center[0], center[1], f"{ind}")


def plot_dsg_objects(
    dsg,
    with_boundary=True,
    plot_indices=False,
    skip_movable=False,
    with_centers=True,
    boundary_lw=1,
):
    if (
        dsg.objects is None
        or dsg.objects.center is None
        or len(dsg.objects.center) == 0
    ):
        return
    p = np.array(dsg.objects.center)
    c = np.array(dsg.objects.semantic_color)
    if with_centers:
        plt.scatter(p[:, 0], p[:, 1], color=c, s=10)

    if with_boundary:
        for b, c, m in zip(
            dsg.objects.box, dsg.objects.semantic_color, dsg.objects.movable
        ):

            if (not skip_movable) or (not m):
                plt.plot(
                    b[:4, 0][[0, 1, 2, 3, 0]],
                    b[:4, 1][[0, 1, 2, 3, 0]],
                    color=c,
                    linewidth=boundary_lw,
                )

    if plot_indices:
        for ind, center in zip(dsg.objects.hydra_symbol, dsg.objects.center):
            plt.text(center[0], center[1], f"{ind}")


def plot_dsg_rooms(
    dsg,
    with_boundary=True,
    plot_indices=True,
    with_edges=True,
    boundary_lw=1,
    edge_lw=1,
):
    if dsg.rooms is None or dsg.rooms.center is None:
        return
    if with_boundary:
        for b, c in zip(dsg.rooms.box, dsg.rooms.semantic_color):
            plt.plot(
                b[:4, 0][[0, 1, 2, 3, 0]],
                b[:4, 1][[0, 1, 2, 3, 0]],
                color=c,
                linewidth=boundary_lw,
            )

    if plot_indices:
        for ind, center in zip(dsg.rooms.hydra_symbol, dsg.rooms.center):
            plt.text(center[0], center[1], f"{ind}")

    if with_edges:
        lines = []
        neighbors = dsg.rooms.sibling_dict
        for node_s in neighbors:
            start = dsg.rooms[node_s].center[:2]
            for node_t in neighbors[node_s]:
                end = dsg.rooms[node_t].center[:2]
                lines.append((start, end))

        lc = LineCollection(lines, linewidth=edge_lw, color="k")
        plt.gca().add_collection(lc)


def plot_plan(start_pos, goal_pos, wps):
    plt.scatter(*start_pos, marker="P", s=600, zorder=10)
    plt.scatter(*goal_pos, marker="*", s=600, zorder=10)
    plt.plot(wps[:, 0], wps[:, 1], color="m")
