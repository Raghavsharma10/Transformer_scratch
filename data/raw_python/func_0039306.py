def get_model_choices():
    """
    Get the select options for the model selector

    :return:
    """
    result = []
    for ct in ContentType.objects.order_by('app_label', 'model'):
        try:
            if issubclass(ct.model_class(), TranslatableModel):
                result.append(
                    ('{} - {}'.format(ct.app_label, ct.model.lower()),
                     '{} - {}'.format(ct.app_label.capitalize(), ct.model_class()._meta.verbose_name_plural))
                )
        except TypeError:
            continue
    return result