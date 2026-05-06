def parse_registries(filesystem, registries):
    """Returns a dictionary with the content of the given registry hives.

    {"\\Registry\\Key\\", (("ValueKey", "ValueType", ValueValue))}

    """
    results = {}

    for path in registries:
        with NamedTemporaryFile(buffering=0) as tempfile:
            filesystem.download(path, tempfile.name)

            registry = RegistryHive(tempfile.name)
            registry.rootkey = registry_root(path)

            results.update({k.path: (k.timestamp, k.values)
                            for k in registry.keys()})

    return results