def get_application_choices():
    """
    Get the select options for the application selector

    :return:
    """
    result = []
    keys = set()
    for ct in ContentType.objects.order_by('app_label', 'model'):
        try:
            if issubclass(ct.model_class(), TranslatableModel) and ct.app_label not in keys:
                result.append(('{}'.format(ct.app_label), '{}'.format(ct.app_label.capitalize())))
                keys.add(ct.app_label)
        except TypeError:
            continue
    return result