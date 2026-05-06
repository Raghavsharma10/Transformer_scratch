def collect_manifest_dependencies(manifest_data, lockfile_data):
    """Convert the manifest format to the dependencies schema"""
    output = {}

    for dependencyName, dependencyConstraint in manifest_data.items():
        output[dependencyName] = {
            # identifies where this dependency is installed from
            'source': 'example-package-manager',
            # the constraint that the user is using (i.e. "> 1.0.0")
            'constraint': dependencyConstraint,
            # all available versions above and outside of their constraint
            # - usually you would need to use the package manager lib or API
            #   to get this information (we just fake it here)
            'available': [
                {'name': '2.0.0'},
            ],
        }

    return output