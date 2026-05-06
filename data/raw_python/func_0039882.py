def get_task_module(feature):
    """
    Return imported task module of feature.

    This function first tries to import the feature and raises FeatureNotFound
    if that is not possible.
    Thereafter, it looks for a submodules called ``apetasks`` and ``tasks`` in that order.
    If such a submodule exists, it is imported and returned.

    :param feature: name of feature to fet task module for.
    :raises: FeatureNotFound if feature_module could not be imported.
    :return: imported module containing the ape tasks of feature or None,
                if module cannot be imported.
    """
    try:
        importlib.import_module(feature)
    except ImportError:
        raise FeatureNotFound(feature)

    tasks_module = None

    # ape tasks may be located in a module called apetasks
    # or (if no apetasks module exists) in a module called tasks
    try:
        tasks_module = importlib.import_module(feature + '.apetasks')
    except ImportError:
        # No apetasks module in feature ... try tasks
        pass

    try:
        tasks_module = importlib.import_module(feature + '.tasks')
    except ImportError:
        # No tasks module in feature ... skip it
        pass

    return tasks_module