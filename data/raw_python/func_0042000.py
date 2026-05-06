def is_table_included(table, names):
    """Determines if the table is included by reference in the names.

    A table can be named by its component or its model (using the short-name
    or a full python path).

    eg. 'package.models.SomeModel' or 'package:SomeModel' or 'package'
        would all include 'SomeModel'.
    """

    # No names indicates that every table is included.
    if not names:
        return True

    # Introspect the table and pull out the model and component from it.
    model, component = table.class_, table.class_._component

    # Check for the component name.
    if component in names:
        return True

    # Check for the full python name.
    model_name = '%s.%s' % (model.__module__, model.__name__)
    if model_name in names:
        return True

    # Check for the short name.
    short_name = '%s:%s' % (component, model.__name__)
    if short_name in names:
        return True

    return False