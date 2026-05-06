def models_from_model(model, include_related=False, exclude=None):
    '''Generator of all model in model.'''
    if exclude is None:
        exclude = set()
    if model and model not in exclude:
        exclude.add(model)
        if isinstance(model, ModelType) and not model._meta.abstract:
            yield model
            if include_related:
                exclude.add(model)
                for field in model._meta.fields:
                    if hasattr(field, 'relmodel'):
                        through = getattr(field, 'through', None)
                        for rmodel in (field.relmodel, field.model, through):
                            for m in models_from_model(
                                    rmodel, include_related=include_related,
                                    exclude=exclude):
                                yield m
                for manytomany in model._meta.manytomany:
                    related = getattr(model, manytomany)
                    for m in models_from_model(related.model,
                                               include_related=include_related,
                                               exclude=exclude):
                        yield m
        elif not isinstance(model, ModelType) and isclass(model):
            # This is a class which is not o ModelType
            yield model