def get_ancestors_of_type(pdsg, node, target_node_type):

    ancestors = set()
    open_set = set()
    closed_set = set()

    open_set.update(pdsg.get_parents(node))
    while len(open_set) > 0:
        n = open_set.pop()
        closed_set.add(n)
        if isinstance(pdsg[n], target_node_type):
            ancestors.add(n)
        open_set.update(pdsg.get_parents(n))
    return ancestors


def get_descendants_of_type(pdsg, node, target_node_type):
    descendants = set()
    open_set = set()
    closed_set = set()

    open_set.update(pdsg.get_children(node))
    while len(open_set) > 0:
        n = open_set.pop()
        closed_set.add(n)
        if isinstance(pdsg[n], target_node_type):
            descendants.add(n)
        open_set.update(pdsg.get_children(n))
    return descendants
