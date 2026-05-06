def get_existing_model(model_name):
    """ Try to find existing model class named `model_name`.

    :param model_name: String name of the model class.
    """
    try:
        model_cls = engine.get_document_cls(model_name)
        log.debug('Model `{}` already exists. Using existing one'.format(
            model_name))
        return model_cls
    except ValueError:
        log.debug('Model `{}` does not exist'.format(model_name))