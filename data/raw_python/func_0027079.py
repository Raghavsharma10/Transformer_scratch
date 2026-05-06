def get_sorted_dependencies(service_model):
    """
    Returns list of application models in topological order.
    It is used in order to correctly delete dependent resources.
    """
    app_models = list(service_model._meta.app_config.get_models())
    dependencies = {model: set() for model in app_models}
    relations = (
        relation
        for model in app_models
        for relation in model._meta.related_objects
        if relation.on_delete in (models.PROTECT, models.CASCADE)
    )
    for rel in relations:
        dependencies[rel.model].add(rel.related_model)
    return stable_topological_sort(app_models, dependencies)