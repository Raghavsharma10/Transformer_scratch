def search_form(*fields, **kwargs):
    """
    Construct a search form filter form using the fields
    provided as arguments to this function.

    By default a field will be created for each field passed
    and hidden field will be created for search. If you pass
    the key work argument `search_only` then only a visible
    search field will be created on the form.

    Passing `status_filter` will include a version status filter
    on this form.
    """

    fdict = {
        'search_fields': set(fields)
    }

    if kwargs.get('search_only'):
        fdict['search'] = forms.CharField(max_length=255, required=False)
    else:
        fdict['search'] = forms.CharField(max_length=255, required=False,
                                          widget=forms.HiddenInput)
        for f in fields:
            fdict[f] = forms.CharField(max_length=255, required=False)

    if kwargs.get('status_filter', False):
        return type("filterform", (VersionFilterForm,), fdict)
    else:
        return type("filterform", (BaseFilterForm,), fdict)