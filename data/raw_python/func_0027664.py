def extract_deps(bundles, log=None):
    """Extract the dependencies from the bundle and its sub-bundles."""
    def _flatten(bundle):
        deps = []
        if hasattr(bundle, 'npm'):
            deps.append(bundle.npm)
        for content in bundle.contents:
            if isinstance(content, BundleBase):
                deps.extend(_flatten(content))
        return deps

    flatten_deps = []
    for bundle in bundles:
        flatten_deps.extend(_flatten(bundle))

    packages = defaultdict(list)
    for dep in flatten_deps:
        for pkg, version in dep.items():
            packages[pkg].append(version)

    deps = {}
    for package, versions in packages.items():
        deps[package] = semver.max_satisfying(versions, '*', True)

        if log and len(versions) > 1:
            log('Warn: {0} version {1} resolved to: {2}'.format(
                repr(package), versions, repr(deps[package])
            ))

    return deps