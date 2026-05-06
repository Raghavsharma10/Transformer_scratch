def makeSoftwareVersion(store, version, systemVersion):
    """
    Return the SoftwareVersion object from store corresponding to the
    version object, creating it if it doesn't already exist.
    """
    return store.findOrCreate(SoftwareVersion,
                              systemVersion=systemVersion,
                              package=unicode(version.package),
                              version=unicode(version.short()),
                              major=version.major,
                              minor=version.minor,
                              micro=version.micro)