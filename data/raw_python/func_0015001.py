def get_form(self, request, obj=None, **kwargs):
    """
    Patched method for PageAdmin.get_form.

    Returns a page form without the base field 'meta_description' which is
    overridden in djangocms-page-meta.

    This is triggered in the page add view and in the change view if
    the meta description of the page is empty.
    """
    language = get_language_from_request(request, obj)
    form = _BASE_PAGEADMIN__GET_FORM(self, request, obj, **kwargs)
    if not obj or not obj.get_meta_description(language=language):
        form.base_fields.pop('meta_description', None)

    return form