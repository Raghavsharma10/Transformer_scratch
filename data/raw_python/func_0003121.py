def install_custom_css( destdir, cssfile, resource=PKGNAME ):
    """
    Add the kernel CSS to custom.css
    """
    ensure_dir_exists( destdir )
    custom = os.path.join( destdir, 'custom.css' )
    prefix = css_frame_prefix(resource)

    # Check if custom.css already includes it. If so, let's remove it first
    exists = False
    if os.path.exists( custom ):
        with io.open(custom) as f:
            for line in f:
                if line.find( prefix ) >= 0:
                    exists = True
                    break
    if exists:
        remove_custom_css( destdir, resource )

    # Fetch the CSS file
    cssfile += '.css'
    data = pkgutil.get_data( resource, os.path.join('resources',cssfile) )
    # get_data() delivers encoded data, str (Python2) or bytes (Python3)

    # Add the CSS at the beginning of custom.css
    # io.open uses unicode strings (unicode in Python2, str in Python3)
    with io.open(custom + '-new', 'wt', encoding='utf-8') as fout:
        fout.write( u'{}START ======================== */\n'.format(prefix))
        fout.write( data.decode('utf-8') )
        fout.write( u'{}END ======================== */\n'.format(prefix))
        if os.path.exists( custom ):
            with io.open( custom, 'rt', encoding='utf-8' ) as fin:
                for line in fin:
                    fout.write( unicode(line) )
    os.rename( custom+'-new',custom)