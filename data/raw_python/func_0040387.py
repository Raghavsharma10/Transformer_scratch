def get_supported_file_loaders_2(force=False):
    """Returns a list of file-based module loaders.
    Each item is a tuple (loader, suffixes).
    """

    if force or (2, 7) <= sys.version_info < (3, 4):  # valid until which py3 version ?

        import imp

        loaders = []
        for suffix, mode, type in imp.get_suffixes():
            if type == imp.PY_SOURCE:
                loaders.append((SourceFileLoader2, [suffix]))
            else:
                loaders.append((ImpFileLoader2, [suffix]))
        return loaders

    elif sys.version_info >= (3, 4):  # valid from which py3 version ?

        from importlib.machinery import (
            SOURCE_SUFFIXES, SourceFileLoader,
            BYTECODE_SUFFIXES, SourcelessFileLoader,
            EXTENSION_SUFFIXES, ExtensionFileLoader,
        )

        # This is already defined in importlib._bootstrap_external
        # but is not exposed.
        extensions = ExtensionFileLoader, EXTENSION_SUFFIXES
        source = SourceFileLoader, SOURCE_SUFFIXES
        bytecode = SourcelessFileLoader, BYTECODE_SUFFIXES
        return [extensions, source, bytecode]