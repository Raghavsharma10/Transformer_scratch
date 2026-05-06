def _get_all_dependencies_of(name, deps=set(), force=False):
    '''Returns list of dependencies of the given dap from Dapi recursively'''
    first_deps = _get_api_dependencies_of(name, force=force)
    for dep in first_deps:
        dep = _strip_version_from_dependency(dep)
        if dep in deps:
            continue
        # we do the following not to resolve the dependencies of already installed daps
        if dap in get_installed_daps():
            continue
        deps |= _get_all_dependencies_of(dep, deps)
    return deps | set([name])