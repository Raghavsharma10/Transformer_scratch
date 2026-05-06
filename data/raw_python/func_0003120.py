def install_kernel_resources( destdir, resource=PKGNAME, files=None ):
    """
    Copy the resource files to the kernelspec folder.
    """
    if files is None:
        files = ['logo-64x64.png', 'logo-32x32.png']
    for filename in files:
        try:
            copyresource( resource, filename, destdir )
        except Exception as e:
            sys.stderr.write(str(e))