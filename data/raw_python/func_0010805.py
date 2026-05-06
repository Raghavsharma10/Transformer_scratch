def add_permission(content_type, permission):
    """
    Adds the passed in permission to that content type.  Note that the permission passed
    in should be a single word, or verb.  The proper 'codename' will be generated from that.
    """
    # build our permission slug
    codename = "%s_%s" % (content_type.model, permission)

    # sys.stderr.write("Checking %s permission for %s\n" % (permission, content_type.name))

    # does it already exist
    if not Permission.objects.filter(content_type=content_type, codename=codename):
        Permission.objects.create(content_type=content_type,
                                  codename=codename,
                                  name="Can %s %s" % (permission, content_type.name))