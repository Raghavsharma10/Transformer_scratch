def get_parent_active_language_choices(parent_object, exclude_current=False):
    """
    .. versionadded:: 1.0

    Get the currently active languages of an parent object.

    Note: if there is no content at the page, the language won't be returned.
    """
    assert parent_object is not None, "Missing parent_object!"

    from .db import ContentItem
    qs = ContentItem.objects \
        .parent(parent_object, limit_parent_language=False) \
        .values_list('language_code', flat=True).distinct()

    languages = set(qs)

    if exclude_current:
        parent_lang = get_parent_language_code(parent_object)
        languages.discard(parent_lang)

    if parler_appsettings.PARLER_LANGUAGES and not parler_appsettings.PARLER_SHOW_EXCLUDED_LANGUAGE_TABS:
        site_id = get_parent_site_id(parent_object)
        try:
            lang_dict = parler_appsettings.PARLER_LANGUAGES[site_id]
        except KeyError:
            lang_dict = ()

        allowed_languages = set(item['code'] for item in lang_dict)
        languages &= allowed_languages

    # No multithreading issue here, object is instantiated for this user only.
    choices = [(lang, str(get_language_title(lang))) for lang in languages if lang]
    choices.sort(key=lambda tup: tup[1])
    return choices