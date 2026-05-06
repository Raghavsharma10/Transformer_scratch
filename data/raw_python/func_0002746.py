def plos_doi_to_xmlurl(doi_string):
    """
    Attempts to resolve a PLoS DOI into a URL path to the XML file.
    """
    #Create URL to request DOI resolution from http://dx.doi.org
    doi_url = 'http://dx.doi.org/{0}'.format(doi_string)
    log.debug('DOI URL: {0}'.format(doi_url))
    #Open the page, follow the redirect
    try:
        resolved_page = urllib.request.urlopen(doi_url)
    except urllib.error.URLError as err:
        print('Unable to resolve DOI URL, or could not connect')
        raise err
    else:
        #Given the redirection, attempt to shape new request for PLoS servers
        resolved_address = resolved_page.geturl()
        log.debug('DOI resolved to {0}'.format(resolved_address))
        parsed = urllib.parse.urlparse(resolved_address)
        xml_url = '{0}://{1}'.format(parsed.scheme, parsed.netloc)
        xml_url += '/article/fetchObjectAttachment.action?uri='
        xml_path = parsed.path.replace(':', '%3A').replace('/', '%2F')
        xml_path = xml_path.split('article%2F')[1]
        xml_url += '{0}{1}'.format(xml_path, '&representation=XML')
        log.debug('Shaped PLoS request for XML {0}'.format(xml_url))
        #Return this url to the calling function
        return xml_url