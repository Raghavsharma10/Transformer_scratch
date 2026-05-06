def encode_network(root):
    """Yield ref-containing obj table entries from object network"""
    def fix_values(obj):
        if isinstance(obj, Container):
            obj.update((k, get_ref(v)) for (k, v) in obj.items()
                                       if k != 'class_name')
            fixed_obj = obj

        elif isinstance(obj, Dictionary):
            fixed_obj = obj.__class__(dict(
                (get_ref(field), get_ref(value))
                for (field, value) in obj.value.items()
            ))

        elif isinstance(obj, dict):
            fixed_obj = dict(
                (get_ref(field), get_ref(value))
                for (field, value) in obj.items()
            )

        elif isinstance(obj, list):
            fixed_obj = [get_ref(field) for field in obj]

        elif isinstance(obj, Form):
            fixed_obj = obj.__class__(**dict(
                (field, get_ref(value))
                for (field, value) in obj.value.items()
            ))

        elif isinstance(obj, ContainsRefs):
            fixed_obj = obj.__class__([get_ref(field)
                                       for field in obj.value])

        else:
            return obj

        fixed_obj._made_from = obj
        return fixed_obj

    objects = []

    def get_ref(obj, objects=objects):
        obj = PythonicAdapter(Pass)._encode(obj, None)

        if isinstance(obj, (FixedObject, Container)):
            if getattr(obj, '_index', None):
                index = obj._index
            else:
                objects.append(None)
                obj._index = index = len(objects)
                objects[index - 1] = fix_values(obj)
            return Ref(index)
        else:
            return obj # Inline value

    get_ref(root)

    for obj in objects:
        if getattr(obj, '_index', None):
            del obj._index
    return objects