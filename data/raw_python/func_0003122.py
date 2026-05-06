def remove_custom_css(destdir, resource=PKGNAME ):
    """
    Remove the kernel CSS from custom.css
    """

    # Remove the inclusion in the main CSS
    if not os.path.isdir( destdir ):
        return False
    custom = os.path.join( destdir, 'custom.css' )
    copy = True
    found = False
    prefix = css_frame_prefix(resource)
    with io.open(custom + '-new', 'wt') as fout:
        with io.open(custom) as fin:
            for line in fin:
                if line.startswith( prefix + 'START' ):
                    copy = False
                    found = True
                elif line.startswith( prefix + 'END' ):
                    copy = True
                elif copy:
                    fout.write( line )

    if found:
        os.rename( custom+'-new',custom)
    else:
        os.unlink( custom+'-new')

    return found