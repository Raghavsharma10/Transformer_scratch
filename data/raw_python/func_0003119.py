def copyresource( resource, filename, destdir ):
    """
    Copy a resource file to a destination
    """
    data = pkgutil.get_data(resource, os.path.join('resources',filename) )
    #log.info( "Installing %s", os.path.join(destdir,filename) )
    with open( os.path.join(destdir,filename), 'wb' ) as fp:
        fp.write(data)