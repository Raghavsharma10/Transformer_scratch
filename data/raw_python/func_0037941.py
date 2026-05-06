def import_class(name):
    """Load class from fully-qualified python module name.

    ex: import_class('bulbs.content.models.Content')
    """

    module, _, klass = name.rpartition('.')
    mod = import_module(module)
    return getattr(mod, klass)