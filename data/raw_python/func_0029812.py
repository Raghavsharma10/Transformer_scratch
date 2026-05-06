def get_or_create(session, model, **kwargs):
    """ Get or create sqlalchemy instance.

    Args:
        session (Sqlalchemy session):
        model (sqlalchemy model):
        kwargs (dict): kwargs to lookup or create instance.

    Returns:
        Tuple: first element is found or created instance, second is boolean - True if instance created,
            False if instance found.
    """
    instance = session.query(model).filter_by(**kwargs).first()
    if instance:
        return instance, False
    else:
        instance = model(**kwargs)
        if 'dataset' in kwargs:
            instance.update_sequence_id(session, kwargs['dataset'])
        session.add(instance)
        session.commit()
        return instance, True