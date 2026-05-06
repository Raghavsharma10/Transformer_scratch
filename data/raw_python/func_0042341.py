def _select_deep_meta(field, depth):
    """
    field = {"name":name, "value":["attribute", "path"]}
    r[field.name]=v[field.value], BUT WE MUST DEAL WITH POSSIBLE LIST IN field.value PATH
    RETURN FUNCTION THAT PERFORMS THE MAPPING
    """
    name = field.name
    if hasattr(field.value, "__call__"):
        try:

            def assign(source, destination):
                destination[name] = field.value(wrap(source))
                return 0, None

            return assign
        except Exception as e:

            def assign(source, destination):
                destination[name] = None
                return 0, None

            return assign

    prefix = field.value[depth : len(field.value) - 1 :]
    if prefix:

        def assign(source, destination):
            for i, f in enumerate(prefix):
                source = source.get(f)
                if source is None:
                    return 0, None
                if is_list(source):
                    return depth + i + 1, source

            f = field.value.last()
            try:
                if not f:  # NO NAME FIELD INDICATES SELECT VALUE
                    destination[name] = source
                else:
                    destination[name] = source.get(f)
            except Exception as e:
                Log.error(
                    "{{value}} does not have {{field}} property",
                    value=source,
                    field=f,
                    cause=e,
                )
            return 0, None

        return assign
    else:
        f = field.value[0]
        if not f:  # NO NAME FIELD INDICATES SELECT VALUE

            def assign(source, destination):
                destination[name] = source
                return 0, None

            return assign
        else:

            def assign(source, destination):
                try:
                    destination[name] = source.get(f)
                except Exception as e:
                    Log.error(
                        "{{value}} does not have {{field}} property",
                        value=source,
                        field=f,
                        cause=e,
                    )
                return 0, None

            return assign