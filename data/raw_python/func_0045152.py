def retrieve_object(model, *args, **kwargs):
    """
    Retrieves a specific object from a given model by primary-key
    lookup, and stores it in a context variable.
    
    Syntax::
    
        {% retrieve_object [app_name].[model_name] [lookup kwargs] as [varname] %}
    
    Example::
    
        {% retrieve_object flatpages.flatpage pk=12 as my_flat_page %}
    
    """
    if len(args) == 1:
        kwargs.update({'pk': args[0]})
    _model = _get_model(model)
    try:
        return _model._default_manager.get(**kwargs)
    except _model.DoesNotExist:
        return ''