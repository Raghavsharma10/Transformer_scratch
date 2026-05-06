def install_import_hook():
    """Installs __import__ hook."""
    saved_import = builtins.__import__
    @functools.wraps(saved_import)
    def import_hook(name, *args, **kwargs):
        if name == 'end':
            process_import()
        end
        return saved_import(name, *args, **kwargs)
    end
    builtins.__import__ = import_hook