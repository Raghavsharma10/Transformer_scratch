def dynamic_load(name):
    """Equivalent of "from X import Y" statement using dot notation to specify
    what to import and return.  For example, foo.bar.thing returns the item
    "thing" in the module "foo.bar" """
    pieces = name.split('.')
    item = pieces[-1]
    mod_name = '.'.join(pieces[:-1])

    mod = __import__(mod_name, globals(), locals(), [item])
    return getattr(mod, item)