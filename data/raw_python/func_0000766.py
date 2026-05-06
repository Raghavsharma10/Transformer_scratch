def generate_rest_view(config, model_cls, attrs=None, es_based=True,
                       attr_view=False, singular=False):
    """ Generate REST view for a model class.

    :param model_cls: Generated DB model class.
    :param attr: List of strings that represent names of view methods, new
        generated view should support. Not supported methods are replaced
        with property that raises AttributeError to display MethodNotAllowed
        error.
    :param es_based: Boolean indicating if generated view should read from
        elasticsearch. If True - collection reads are performed from
        elasticsearch. Database is used for reads otherwise.
        Defaults to True.
    :param attr_view: Boolean indicating if ItemAttributeView should be
        used as a base class for generated view.
    :param singular: Boolean indicating if ItemSingularView should be
        used as a base class for generated view.
    """
    valid_attrs = (list(collection_methods.values()) +
                   list(item_methods.values()))
    missing_attrs = set(valid_attrs) - set(attrs)

    if singular:
        bases = [ItemSingularView]
    elif attr_view:
        bases = [ItemAttributeView]
    elif es_based:
        bases = [ESCollectionView]
    else:
        bases = [CollectionView]

    if config.registry.database_acls:
        from nefertari_guards.view import ACLFilterViewMixin
        bases = [SetObjectACLMixin] + bases + [ACLFilterViewMixin]
    bases.append(NefertariBaseView)

    RESTView = type('RESTView', tuple(bases), {'Model': model_cls})

    def _attr_error(*args, **kwargs):
        raise AttributeError

    for attr in missing_attrs:
        setattr(RESTView, attr, property(_attr_error))

    return RESTView