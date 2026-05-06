def confirmRelMe(profileURL, resourceURL, profileRelMes=None, resourceRelMes=None):
    """Determine if a given :resourceURL: is authoritative for the :profileURL:

    TODO add https/http filtering for those who wish to limit/restrict urls to match fully
    TODO add code to ensure that each item in the redirect chain is authoritative

    :param profileURL: URL of the user
    :param resourceURL: URL of the resource to validate
    :param profileRelMes: optional list of rel="me" links within the profile URL
    :param resourceRelMes: optional list of rel="me" links found within resource URL
    :rtype: True if confirmed
    """
    result  = False
    profile = normalizeURL(profileURL)

    if profileRelMes is None:
        profileRelMe = findRelMe(profileURL)
        profileRelMes = profileRelMe['relme']
    if resourceRelMes is None:
        resourceRelMe = findRelMe(resourceURL)
        resourceRelMes = resourceRelMe['relme']

    for url in resourceRelMes:
        if profile in (url, normalizeURL(url)):
            result = True
            break

    return result