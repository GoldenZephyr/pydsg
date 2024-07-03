from pydsg import *
from pydsg_translator import spark_dsg_to_pydsg, load_semantic_id_map_from_label_space
import spark_dsg
from dsg_plotter import plot_dsg_places, plot_dsg_rooms, plot_dsg_objects
import matplotlib.pyplot as plt

dsg_fn = "/home/ubuntu/catkin_ws/src/active_dsg_launch/dsg.json"

G = spark_dsg.DynamicSceneGraph.load(dsg_fn)

labelspace_path = (
    "/home/ubuntu/catkin_ws/src/hydra/config/label_spaces/ade20k_full_label_space.yaml"
)
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
pdsg = spark_dsg_to_pydsg(
    G,
    semantic_id_to_label=semantic_id_to_label,
    traversable_semantics=traversable_semantics,
)

plot_dsg_places(pdsg)
plot_dsg_objects(pdsg)
plt.show()

pdsg.save("dsg_save_test")

pdsgl = PyDSG.load("dsg_save_test")
