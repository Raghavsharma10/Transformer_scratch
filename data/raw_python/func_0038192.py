def resolve_labels(model_label):
    """
    Seperate model_label into parts.
    Returns dictionary with app, model and app_model strings.
    """
    labels = {}

    # Resolve app label.
    labels['app'] = model_label.split('.')[0]

    # Resolve model label
    labels['model'] = model_label.split('.')[-1]

    # Resolve module_app_model label.
    labels['app_model'] = '%s.%s' % (labels['app'], labels['model'])

    return labels