def require_version(namespace, version):
    """Set a version for the namespace to be loaded.
    This needs to be called before importing the namespace or any
    namespace that depends on it.
    """

    global _versions

    repo = GIRepository()
    namespaces = repo.get_loaded_namespaces()

    if namespace in namespaces:
        loaded_version = repo.get_version(namespace)
        if loaded_version != version:
            raise ValueError('Namespace %s is already loaded with version %s' %
                             (namespace, loaded_version))

    if namespace in _versions and _versions[namespace] != version:
        raise ValueError('Namespace %s already requires version %s' %
                         (namespace, _versions[namespace]))

    available_versions = repo.enumerate_versions(namespace)
    if not available_versions:
        raise ValueError('Namespace %s not available' % namespace)

    if version not in available_versions:
        raise ValueError('Namespace %s not available for version %s' %
                         (namespace, version))

    _versions[namespace] = version