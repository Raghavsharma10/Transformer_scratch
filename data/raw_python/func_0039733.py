def collect_lockfile_dependencies(lockfile_data):
    """Convert the lockfile format to the dependencies schema"""
    output = {}

    for dependencyName, installedVersion in lockfile_data.items():
        output[dependencyName] = {
            'source': 'example-package-manager',
            'installed': {'name': installedVersion},
        }

    return output