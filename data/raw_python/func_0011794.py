def _discover_cover_image(zf, opf_xmldoc, opf_filepath):
    '''
    Find the cover image path in the OPF file.
    Returns a tuple: (image content in base64, file extension)
    '''
    content = None
    filepath = None
    extension = None

    # Strategies to discover the cover-image path:

    # e.g.: <meta name="cover" content="cover"/>
    tag = find_tag(opf_xmldoc, 'meta', 'name', 'cover')
    if tag and 'content' in tag.attributes.keys():
        item_id = tag.attributes['content'].value
        if item_id:
            # e.g.: <item href="cover.jpg" id="cover" media-type="image/jpeg"/>
            filepath, extension = find_img_tag(opf_xmldoc, 'item', 'id', item_id)
    if not filepath:
        filepath, extension = find_img_tag(opf_xmldoc, 'item', 'id', 'cover-image')
    if not filepath:
        filepath, extension = find_img_tag(opf_xmldoc, 'item', 'id', 'cover')

    # If we have found the cover image path:
    if filepath:
        # The cover image path is relative to the OPF file
        base_dir = os.path.dirname(opf_filepath)
        # Also, normalize the path (ie opfpath/../cover.jpg -> cover.jpg)
        coverpath = os.path.normpath(os.path.join(base_dir, filepath))
        content = zf.read(coverpath)
        content = base64.b64encode(content)

    return content, extension