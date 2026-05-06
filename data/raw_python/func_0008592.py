def decode_network(objects):
    """Return root object from ref-containing obj table entries"""
    def resolve_ref(obj, objects=objects):
        if isinstance(obj, Ref):
            # first entry is 1
            return objects[obj.index - 1]
        else:
            return obj

    # Reading the ObjTable backwards somehow makes more sense.
    for i in xrange(len(objects)-1, -1, -1):
        obj = objects[i]

        if isinstance(obj, Container):
            obj.update((k, resolve_ref(v)) for (k, v) in obj.items())

        elif isinstance(obj, Dictionary):
            obj.value = dict(
                (resolve_ref(field), resolve_ref(value))
                for (field, value) in obj.value.items()
            )

        elif isinstance(obj, dict):
            obj = dict(
                (resolve_ref(field), resolve_ref(value))
                for (field, value) in obj.items()
            )

        elif isinstance(obj, list):
            obj = [resolve_ref(field) for field in obj]

        elif isinstance(obj, Form):
            for field in obj.value:
                value = getattr(obj, field)
                value = resolve_ref(value)
                setattr(obj, field, value)

        elif isinstance(obj, ContainsRefs):
            obj.value = [resolve_ref(field) for field in obj.value]

        objects[i] = obj

    for obj in objects:
        if isinstance(obj, Form):
            obj.built()

    root = objects[0]
    return root