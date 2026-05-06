def to_safe_str(s):
    """
    converts some (tr) non-ascii chars to ascii counterparts,
    then return the result as lowercase
    """
    # TODO: This is insufficient as it doesn't do anything for other non-ascii chars
    return re.sub(r'[^0-9a-zA-Z]+', '_', s.strip().replace(u'ğ', 'g').replace(u'ö', 'o').replace(
        u'ç', 'c').replace(u'Ç','c').replace(u'Ö', u'O').replace(u'Ş', 's').replace(
        u'Ü', 'u').replace(u'ı', 'i').replace(u'İ','i').replace(u'Ğ', 'g').replace(
        u'ö', 'o').replace(u'ş', 's').replace(u'ü', 'u').lower(), re.UNICODE)