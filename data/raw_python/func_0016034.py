def pdf_doc_info(instance):
    """Ensure the keys of the 'document_info_dict' property of the pdf-ext
    extension of file objects are only valid PDF Document Information
    Dictionary Keys.
    """
    for key, obj in instance['objects'].items():
        if ('type' in obj and obj['type'] == 'file'):
            try:
                did = obj['extensions']['pdf-ext']['document_info_dict']
            except KeyError:
                continue

            for elem in did:
                if elem not in enums.PDF_DID:
                    yield JSONError("The 'document_info_dict' property of "
                                    "object '%s' contains a key ('%s') that is"
                                    " not a valid PDF Document Information "
                                    "Dictionary key."
                                    % (key, elem), instance['id'],
                                    'pdf-doc-info')