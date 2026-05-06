def get_epub_opf_xml(filepath):
    '''
    Returns the file.OPF contents of the ePub file
    '''
    if not zipfile.is_zipfile(filepath):
        raise EPubException('Unknown file')

    # print('Reading ePub file: {}'.format(filepath))
    zf = zipfile.ZipFile(filepath, 'r', compression=zipfile.ZIP_DEFLATED, allowZip64=True)
    container = zf.read('META-INF/container.xml')
    container_xmldoc = minidom.parseString(container)
    # e.g.: <rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>
    opf_filepath = container_xmldoc.getElementsByTagName('rootfile')[0].attributes['full-path'].value
    return zf.read(opf_filepath)