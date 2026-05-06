def _check_require_version(namespace, stacklevel):
    """A context manager which tries to give helpful warnings
    about missing gi.require_version() which could potentially
    break code if only an older version than expected is installed
    or a new version gets introduced.

    ::

        with _check_require_version("Gtk", stacklevel):
            load_namespace_and_overrides()
    """

    repository = GIRepository()
    was_loaded = repository.is_registered(namespace)

    yield

    if was_loaded:
        # it was loaded before by another import which depended on this
        # namespace or by C code like libpeas
        return

    if namespace in ("GLib", "GObject", "Gio"):
        # part of glib (we have bigger problems if versions change there)
        return

    if get_required_version(namespace) is not None:
        # the version was forced using require_version()
        return

    version = repository.get_version(namespace)
    warnings.warn(
        "%(namespace)s was imported without specifying a version first. "
        "Use gi.require_version('%(namespace)s', '%(version)s') before "
        "import to ensure that the right version gets loaded."
        % {"namespace": namespace, "version": version},
        PyGIWarning, stacklevel=stacklevel)