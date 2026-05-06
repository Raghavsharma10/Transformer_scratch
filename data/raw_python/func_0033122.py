def get_files_for_document(document):
    """
    Returns the available files for all languages.

    In case the file is already present in another language, it does not re-add
    it again.

    """
    files = []
    for doc_trans in document.translations.all():
        if doc_trans.filer_file is not None and \
                doc_trans.filer_file not in files:
            doc_trans.filer_file.language = doc_trans.language_code
            files.append(doc_trans.filer_file)
    return files