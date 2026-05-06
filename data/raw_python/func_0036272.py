def alphabet(cl):
    """
    The inclusion tag that renders the admin/alphabet.html template in the
    admin. Accepts a ChangeList object, which is custom to the admin.
    """
    if not getattr(cl.model_admin, 'alphabet_filter', False):
        return
    field_name = cl.model_admin.alphabet_filter
    alpha_field = '%s__istartswith' % field_name
    alpha_lookup = cl.params.get(alpha_field, '')

    letters_used = _get_available_letters(field_name, cl.model.objects.all())
    all_letters = list(_get_default_letters(cl.model_admin) | letters_used)
    all_letters.sort()

    choices = [{
        'link': cl.get_query_string({alpha_field: letter}),
        'title': letter,
        'active': letter == alpha_lookup,
        'has_entries': letter in letters_used, } for letter in all_letters]
    all_letters = [{
        'link': cl.get_query_string(None, [alpha_field]),
        'title': _('All'),
        'active': '' == alpha_lookup,
        'has_entries': True
    }, ]
    return {'choices': all_letters + choices}