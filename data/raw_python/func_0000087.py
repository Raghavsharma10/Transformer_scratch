def import_transformer(name):
    '''If needed, import a transformer, and adds it to the globally known dict
       The code inside a module where a transformer is defined should be
       standard Python code, which does not need any transformation.
       So, we disable the import hook, and let the normal module import
       do its job - which is faster and likely more reliable than our
       custom method.
    '''
    if name in transformers:
        return transformers[name]

    # We are adding a transformer built from normal/standard Python code.
    # As we are not performing transformations, we temporarily disable
    # our import hook, both to avoid potential problems AND because we
    # found that this resulted in much faster code.
    hook = sys.meta_path[0]
    sys.meta_path = sys.meta_path[1:]
    try:
        transformers[name] = __import__(name)
        # Some transformers are not allowed in the console.
        # If an attempt is made to activate one of them in the console,
        # we replace it by a transformer that does nothing and print a
        # message specific to that transformer as written in its module.
        if CONSOLE_ACTIVE:
            if hasattr(transformers[name], "NO_CONSOLE"):
                print(transformers[name].NO_CONSOLE)
                transformers[name] = NullTransformer()
    except ImportError:
        sys.stderr.write("Warning: Import Error in add_transformers: %s not found\n" % name)
        transformers[name] = NullTransformer()
    except Exception as e:
        sys.stderr.write("Unexpected exception in transforms.import_transformer%s\n " %
                         e.__class__.__name__)
    finally:
        sys.meta_path.insert(0, hook) # restore import hook

    return transformers[name]