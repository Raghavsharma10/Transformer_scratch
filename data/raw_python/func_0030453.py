def qualified_name_import(cls):
    """Full name of a class, including the module. Like qualified_class_name, but when you already have a class """

    parts = qualified_name(cls).split('.')

    return "from {} import {}".format('.'.join(parts[:-1]), parts[-1])