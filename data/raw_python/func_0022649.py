def xym(source_id, srcdir, dstdir, strict=False, strict_examples=False, debug_level=0, add_line_refs=False,
        force_revision_pyang=False, force_revision_regexp=False):
    """
    Extracts YANG model from an IETF RFC or draft text file.
    This is the main (external) API entry for the module.

    :param add_line_refs:
    :param source_id: identifier (file name or URL) of a draft or RFC file containing
           one or more YANG models
    :param srcdir: If source_id points to a file, the optional parameter identifies
           the directory where the file is located
    :param dstdir: Directory where to put the extracted YANG models
    :param strict: Strict syntax enforcement
    :param strict_examples: Only output valid examples when in strict mode
    :param debug_level: Determines how much debug output is printed to the console
    :param force_revision_regexp: Whether it should create a <filename>@<revision>.yang even on error using regexp
    :param force_revision_pyang: Whether it should create a <filename>@<revision>.yang even on error using pyang
    :return: None
    """

    if force_revision_regexp and force_revision_pyang:
        print('Can not use both methods for parsing name and revision - using regular expression method only')
        force_revision_pyang = False

    url = re.compile(r'^(?:http|ftp)s?://'  # http:// or https://
                     r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain
                     r'localhost|'  # localhost...
                     r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
                     r'(?::\d+)?'  # optional port
                     r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    rqst_hdrs = {'Accept': 'text/plain', 'Accept-Charset': 'utf-8'}

    ye = YangModuleExtractor(source_id, dstdir, strict, strict_examples, add_line_refs, debug_level)
    is_url = url.match(source_id)
    if is_url:
        r = requests.get(source_id, headers=rqst_hdrs)
        if r.status_code == 200:
            content = r.text.encode('utf8').splitlines(True)
            ye.extract_yang_model(content)
        else:
            print("Failed to fetch file from URL '%s', error '%d'" % (source_id, r.status_code), file=sys.stderr)
    else:
        try:
            with open(os.path.join(srcdir, source_id)) as sf:
                ye.extract_yang_model(sf.readlines())
        except IOError as ioe:
            print(ioe)
    return ye.get_extracted_models(force_revision_pyang, force_revision_regexp)