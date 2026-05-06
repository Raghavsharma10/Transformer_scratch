def flex_update_obj(source, target, __silent, *fields, **field_map):
    ''' Pull data from source to target.
    Target's __dict__ (object data) will be used by default. Otherwise, it'll be treated as a dictionary '''
    source_dict = source.__dict__ if hasattr(source, '__dict__') else source
    if not fields:
        fields = source_dict.keys()
    for f in fields:
        if f not in source_dict and __silent:
            continue
        target_f = f if f not in field_map else field_map[f]
        setattr(target, target_f, source_dict[f])