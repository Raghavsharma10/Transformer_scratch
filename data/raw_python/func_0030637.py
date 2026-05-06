def import_class_by_string(name):
    """Return a class by importing its module from a fully qualified string."""
    components = name.split('.')
    clazz = components.pop()
    mod = __import__('.'.join(components))

    components += [clazz]
    for comp in components[1:]:
        mod = getattr(mod, comp)

    return mod