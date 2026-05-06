def replicate_filter(sources, model, cache=None):
    '''Replicates the list of objects to other class and returns their
    reflections'''
    targets = [replicate_no_merge(source, model, cache=cache)
               for source in sources]
    # Some objects may not be available in target DB (not published), so we
    # have to exclude None from the list.
    return [target for target in targets if target is not None]