def observable_object_references(instance):
    """Ensure certain observable object properties reference the correct type
    of object.
    """
    for key, obj in instance['objects'].items():
        if 'type' not in obj:
            continue
        elif obj['type'] not in enums.OBSERVABLE_PROP_REFS:
            continue

        obj_type = obj['type']
        for obj_prop in enums.OBSERVABLE_PROP_REFS[obj_type]:
            if obj_prop not in obj:
                continue
            enum_prop = enums.OBSERVABLE_PROP_REFS[obj_type][obj_prop]
            if isinstance(enum_prop, list):
                refs = obj[obj_prop]
                enum_vals = enum_prop
                for x in check_observable_refs(refs, obj_prop, enum_prop, '',
                                               enum_vals, key, instance):
                    yield x

            elif isinstance(enum_prop, dict):
                for embedded_prop in enum_prop:
                    if isinstance(obj[obj_prop], dict):
                        if embedded_prop not in obj[obj_prop]:
                            continue
                        embedded_obj = obj[obj_prop][embedded_prop]
                        for embed_obj_prop in embedded_obj:
                            if embed_obj_prop not in enum_prop[embedded_prop]:
                                continue
                            refs = embedded_obj[embed_obj_prop]
                            enum_vals = enum_prop[embedded_prop][embed_obj_prop]
                            for x in check_observable_refs(refs, obj_prop, enum_prop,
                                                           embed_obj_prop, enum_vals,
                                                           key, instance):
                                yield x

                    elif isinstance(obj[obj_prop], list):
                        for embedded_list_obj in obj[obj_prop]:

                            if embedded_prop not in embedded_list_obj:
                                continue
                            embedded_obj = embedded_list_obj[embedded_prop]
                            refs = embedded_obj
                            enum_vals = enum_prop[embedded_prop]
                            for x in check_observable_refs(refs, obj_prop, enum_prop,
                                                           embedded_prop, enum_vals,
                                                           key, instance):
                                yield x