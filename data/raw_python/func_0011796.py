def get_epub_metadata(filepath, read_cover_image=True, read_toc=True):
    '''
    References: http://idpf.org/epub/201 and http://idpf.org/epub/301
    1. Parse META-INF/container.xml file and find the .OPF file path.
    2. In the .OPF file, find the metadata
    '''
    if not zipfile.is_zipfile(filepath):
        raise EPubException('Unknown file')

    # print('Reading ePub file: {}'.format(filepath))
    zf = zipfile.ZipFile(filepath, 'r', compression=zipfile.ZIP_DEFLATED, allowZip64=True)
    container = zf.read('META-INF/container.xml')
    container_xmldoc = minidom.parseString(container)
    # e.g.: <rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>
    opf_filepath = container_xmldoc.getElementsByTagName('rootfile')[0].attributes['full-path'].value

    opf = zf.read(os.path.normpath(opf_filepath))
    opf_xmldoc = minidom.parseString(opf)

    # This file is specific to the authors if it exists.
    authors_html = None
    try:
        authors_html = minidom.parseString(zf.read('OEBPS/pr02.html'))
    except KeyError:
        # Most books store authors using epub tags, so no worries.
        pass

    # This file is specific to the publish date if it exists.
    publish_date_html = None
    try:
        publish_date_html = minidom.parseString(zf.read('OEBPS/pr01.html'))
    except KeyError:
        # Most books store authors using epub tags, so no worries.
        pass


    file_size_in_bytes = os.path.getsize(filepath)

    data = odict({
        'epub_version': _discover_epub_version(opf_xmldoc),
        'title': _discover_title(opf_xmldoc),
        'language': _discover_language(opf_xmldoc),
        'description': _discover_description(opf_xmldoc),
        'authors': _discover_authors(opf_xmldoc, authors_html=authors_html),
        'publisher': _discover_publisher(opf_xmldoc),
        'publication_date': _discover_publication_date(opf_xmldoc,
                                                       date_html=publish_date_html),
        'identifiers': _discover_identifiers(opf_xmldoc),
        'subject': _discover_subject(opf_xmldoc),
        'file_size_in_bytes': file_size_in_bytes,
    })

    if read_cover_image:
        cover_image_content, cover_image_extension = _discover_cover_image(zf, opf_xmldoc, opf_filepath)
        data.cover_image_content = cover_image_content
        data.cover_image_extension = cover_image_extension

    if read_toc:
        data.toc = _discover_toc(zf, opf_xmldoc, opf_filepath)

    return data