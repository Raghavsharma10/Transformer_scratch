def lookup_field(key, lookup_type=None, placeholder=None, html_class="div",
                 select_type="strapselect", mapping="uuid"):
    """Generates a lookup field for form definitions"""

    if lookup_type is None:
        lookup_type = key

    if placeholder is None:
        placeholder = "Select a " + lookup_type

    result = {
        'key': key,
        'htmlClass': html_class,
        'type': select_type,
        'placeholder': placeholder,
        'options': {
            "type": lookup_type,
            "asyncCallback": "$ctrl.getFormData",
            "map": {'valueProperty': mapping, 'nameProperty': 'name'}
        }
    }

    return result