def delete_metric(name):
    """Remove the named metric"""

    with LOCK:
        old_metric = REGISTRY.pop(name, None)

        # look for the metric name in the tags and remove it
        for _, tags in py3comp.iteritems(TAGS):
            if name in tags:
                tags.remove(name)

    return old_metric