def core_choice_fields(metadata_class):
    """ If the 'optional' core fields (_site and _language) are required, 
        list them here. 
    """
    fields = []
    if metadata_class._meta.use_sites:
        fields.append('_site')
    if metadata_class._meta.use_i18n:
        fields.append('_language')
    return fields