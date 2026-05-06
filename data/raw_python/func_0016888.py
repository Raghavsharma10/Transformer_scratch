def structured_storage(filename):
    """Pick out info from MS documents with embedded
     structured storage(typically MS Word docs etc.)

    Returns a dictionary of information found
    """

    if not pythoncom.StgIsStorageFile(filename):
        return {}

    flags = storagecon.STGM_READ | storagecon.STGM_SHARE_EXCLUSIVE
    storage = pythoncom.StgOpenStorage(filename, None, flags)
    try:
        properties_storage = storage.QueryInterface(pythoncom.IID_IPropertySetStorage)
    except pythoncom.com_error:
        return {}

    property_sheet = properties_storage.Open(FMTID_USER_DEFINED_PROPERTIES)
    try:
        data = property_sheet.ReadMultiple(PROPERTIES)
    finally:
        property_sheet = None

    title, subject, author, created_on, keywords, comments, template_used, \
     updated_by, edited_on, printed_on, saved_on, \
     n_pages, n_words, n_characters, \
     application = data

    result = {}
    if title:
        result['title'] = title
    if subject:
        result['subject'] = subject
    if author:
        result['author'] = author
    if created_on:
        result['created_on'] = created_on
    if keywords:
        result['keywords'] = keywords
    if comments:
        result['comments'] = comments
    if template_used:
        result['template_used'] = template_used
    if updated_by:
        result['updated_by'] = updated_by
    if edited_on:
        result['edited_on'] = edited_on
    if printed_on:
        result['printed_on'] = printed_on
    if saved_on:
        result['saved_on'] = saved_on
    if n_pages:
        result['n_pages'] = n_pages
    if n_words:
        result['n_words'] = n_words
    if n_characters:
        result['n_characters'] = n_characters
    if application:
        result['application'] = application
    return result