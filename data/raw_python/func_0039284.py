def show_list_translations(context, item):
    """
    Return the widget to select the translations we want
    to order or delete from the item it's being edited

    :param context:
    :param item:
    :return:
    """
    if not item:
        return

    manager = Manager()
    manager.set_master(item)

    ct_item = ContentType.objects.get_for_model(item)

    item_language_codes = manager.get_languages_from_item(ct_item, item)
    model_language_codes = manager.get_languages_from_model(ct_item.app_label, ct_item.model)

    item_languages = [{'lang': lang, 'from_model': lang.code in model_language_codes}
                      for lang in TransLanguage.objects.filter(code__in=item_language_codes).order_by('name')]

    more_languages = [{'lang': lang, 'from_model': lang.code in model_language_codes}
                      for lang in TransLanguage.objects.exclude(main_language=True).order_by('name')]

    return render_to_string('languages/translation_language_selector.html', {
        'item_languages': item_languages,
        'more_languages': more_languages,
        'api_url': TM_API_URL,
        'app_label': manager.app_label,
        'model': manager.model_label,
        'object_pk': item.pk
    })