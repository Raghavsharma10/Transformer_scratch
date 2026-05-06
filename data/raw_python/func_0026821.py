def _is_custom_qs_manager(funcdef):
    """Checks if a function definition is a queryset manager created
    with the @queryset_manager decorator."""

    decors = getattr(funcdef, 'decorators', None)
    if decors:
        for dec in decors.get_children():
            try:
                if dec.name == 'queryset_manager':  # pragma no branch
                    return True
            except AttributeError:
                continue

    return False