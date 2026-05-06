def encode_network(root):
    """Yield ref-containing obj table entries from object network"""
    orig_objects = []
    objects = []

    def get_ref(value, objects=objects):
        """Returns the index of the given object in the object table,
        adding it if needed.

        """
        value = PythonicAdapter(Pass)._encode(value, None)
        # Convert strs to FixedObjects here to make sure they get encoded
        # correctly

        if isinstance(value, (Container, FixedObject)):
            if getattr(value, '_tmp_index', None):
                index = value._tmp_index
            else:
                objects.append(value)
                index = len(objects)
                value._tmp_index = index
                orig_objects.append(value) # save the object so we can
                                           # strip the _tmp_indexes later
            return Ref(index)
        else:
            return value # Inline value

    def fix_fields(obj):
        obj = PythonicAdapter(Pass)._encode(obj, None)
        # Convert strs to FixedObjects here to make sure they get encoded
        # correctly

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

    root = PythonicAdapter(Pass)._encode(root, None)

    i = 0
    objects = [root]
    root._tmp_index = 1
    while i < len(objects):
        objects[i] = fix_fields(objects[i])
        i += 1

    for obj in orig_objects:
        obj._tmp_index = None
        # Strip indexes off objects in case we save again later

    return objects