import dataclasses
import numpy as np


class StructAppend:
    """Append a struct to a struct of arrays"""

    # TODO: test to see if this actually works
    def append(self, instance):
        for field in dataclasses.fields(self):
            if issubclass(field.type, np.ndarray):
                new_val = getattr(instance, field.name)
                new_val.shape = (1,) + new_val.shape
                setattr(
                    self,
                    field.name,
                    np.concatenate((getattr(self, field.name), new_val)),
                )
            elif issubclass(field.type, list):
                new_val = getattr(instance, field.name)
                setattr(self, field.name, setattr(self, field.name) + [new_val])
        # TODO: refactor this so can be used more generally
        self.update_local_hydra_mapping()


class SoAIndexing:
    """Indexing into struct of arrays

    Giving a single index returns a struct corresponding to the single element selected.
    Also supports slicing / boolean selection, which returns a filtered struct of arrays
    """

    def __getitem__(self, key):
        element_struct_type = self.element_struct_type
        if isinstance(key, slice):
            args = {}

            for field in dataclasses.fields(self):
                should_filter = field.type in [np.ndarray, list, tuple]
                if should_filter:
                    args[field.name] = getattr(self, field.name)[key]
                else:
                    args[field.name] = getattr(self, field.name)

            return type(self)(**args)
        elif isinstance(key, list) or isinstance(key, np.ndarray):
            elements = [e for e, k in zip(self, key) if k]
            element_fields = [
                field.name for field in dataclasses.fields(self.element_struct_type)
            ]
            extra = {
                field.name: getattr(self, field.name)
                for field in dataclasses.fields(self)
                if field.name not in element_fields
            }
            new_soa = to_soa(elements, type(self), extra)
            new_soa.cleanup()
            return new_soa
        else:
            args = {}
            for field in dataclasses.fields(element_struct_type):
                args[field.name] = getattr(self, field.name)[key]
            return element_struct_type(**args)


def to_soa(instances, soa_class, args=None):
    """Construct struct of arrays given iterable of instances"""
    if args is None:
        args = {}
    for field in dataclasses.fields(soa_class):
        if field.name in args:
            continue
        should_aggregate = any(
            issubclass(field.type, t) for t in [np.ndarray, list, tuple, dict]
        )

        aggregation = [
            getattr(inst, field.name) for inst in instances if hasattr(inst, field.name)
        ]

        if not should_aggregate or len(aggregation) == 0:
            if (
                type(field.default) == dataclasses._MISSING_TYPE
                and type(field.default_factory) == dataclasses._MISSING_TYPE
            ):
                try:
                    args[field.name] = field.type()
                except TypeError:
                    args[field.name] = None
            continue
        if field.type == np.ndarray:
            if issubclass(type(aggregation[0]), np.ndarray):
                shapes = [a.shape for a in aggregation]
                if all([s == shapes[0] for s in shapes]):
                    args[field.name] = np.array(aggregation)
                else:
                    args[field.name] = np.array(aggregation, dtype=object)
            elif issubclass(type(aggregation[0]), dict):
                d = aggregation[0]
                for e in aggregation:
                    d = {**d, **e}
                args[field.name] = d
            else:
                args[field.name] = np.array(aggregation)
        else:
            args[field.name] = aggregation

    return soa_class(**args)
