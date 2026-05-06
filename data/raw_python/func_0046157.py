def pull_parent(collector, image, **kwargs):
    """DEPRECATED - use pull_dependencies instead"""
    log.warning("DEPRECATED - use pull_dependencies instead")
    pull_dependencies(collector, image, **kwargs)