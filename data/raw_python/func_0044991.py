def userForCert(store, cert):
    """Gets the user for the given certificate.

    """
    return store.findUnique(User, User.email == emailForCert(cert))