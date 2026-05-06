def doi_input(doi_string, download=True):
    """
    This method accepts a DOI string and attempts to download the appropriate
    xml file. If successful, it returns a path to that file. As with all URL
    input types, the success of this method depends on supporting per-publisher
    conventions and will fail on unsupported publishers
    """
    log.debug('DOI Input - {0}'.format(doi_string))
    doi_string = doi_string[4:]
    if '10.1371' in doi_string:  # Corresponds to PLoS
        log.debug('DOI string shows PLoS')
        xml_url = plos_doi_to_xmlurl(doi_string)
    else:
        log.critical('DOI input for this publisher is not supported')
        sys.exit('This publisher is not yet supported by OpenAccess_EPUB')
    return url_input(xml_url, download)