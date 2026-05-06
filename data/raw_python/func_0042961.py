def _DataClean_(fieldVal):
    """
    Return 'cleaned' value to standardize lookups (convert to uppercase, remove leading/trailing whitespace, carriage returns, line breaks, and unprintable characters)

    :param string fieldVal: field value
    """

    fieldValNew = fieldVal

    fieldValNew = fieldValNew.upper()

    fieldValNew = fieldValNew.strip()

    fieldValNew = re.sub("[\s\n\t]+", " ", fieldValNew)

    return fieldValNew