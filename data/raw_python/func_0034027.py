def check_api_key(request, key, hproPk):
    """Check if an API key is valid"""

    if settings.PIAPI_STANDALONE:
        return True

    (_, _, hproject) = getPlugItObject(hproPk)

    if not hproject:
        return False

    if hproject.plugItApiKey is None or hproject.plugItApiKey == '':
        return False

    return hproject.plugItApiKey == key