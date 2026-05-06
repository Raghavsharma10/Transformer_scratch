def url_input(url_string, download=True):
    """
    This method expects a direct URL link to an xml file. It will apply no
    modifications to the received URL string, so ensure good input.
    """
    log.debug('URL Input - {0}'.format(url_string))
    try:
        open_xml = urllib.request.urlopen(url_string)
    except urllib.error.URLError as err:
        print('utils.input.url_input received a bad URL, or could not connect')
        raise err
    else:
        #Employ a quick check on the mimetype of the link
        if not open_xml.headers['Content-Type'] == 'text/xml':
            sys.exit('URL request does not appear to be XML')
        filename = open_xml.headers['Content-Disposition'].split('\"')[1]
        if download:
            with open(filename, 'wb') as xml_file:
                xml_file.write(open_xml.read())
        return openaccess_epub.utils.file_root_name(filename)