def render_placeholder_search_text(placeholder, fallback_language=None):
    """
    Render a :class:`~fluent_contents.models.Placeholder` object to search text.
    This text can be used by an indexer (e.g. haystack) to produce content search for a parent object.

    :param placeholder: The placeholder object.
    :type placeholder: :class:`~fluent_contents.models.Placeholder`
    :param fallback_language: The fallback language to use if there are no items in the current language.
                              Passing ``True`` uses the default :ref:`FLUENT_CONTENTS_DEFAULT_LANGUAGE_CODE`.
    :type fallback_language: bool|str
    :rtype: str
    """
    parent_object = placeholder.parent   # this is a cached lookup thanks to PlaceholderFieldDescriptor
    language = get_parent_language_code(parent_object)
    output = SearchRenderingPipe(language).render_placeholder(
        placeholder=placeholder,
        parent_object=parent_object,
        fallback_language=fallback_language
    )
    return output.html