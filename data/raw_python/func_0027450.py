def createCertRequest(pkey, digest="sha256"):
    """
    Create a certificate request.
    Arguments: pkey   - The key to associate with the request
               digest - Digestion method to use for signing, default is sha256
               **name - The name of the subject of the request, possible
                        arguments are:
                          C     - Country name
                          ST    - State or province name
                          L     - Locality name
                          O     - Organization name
                          OU    - Organizational unit name
                          CN    - Common name
                          emailAddress - E-mail address
    Returns:   The certificate request in an X509Req object
    """
    req = crypto.X509Req()

    req.get_subject().C = "FR"
    req.get_subject().ST = "IDF"
    req.get_subject().L = "Paris"
    req.get_subject().O = "RedHat"  # noqa
    req.get_subject().OU = "DCI"
    req.get_subject().CN = "DCI-remoteCI"

    req.set_pubkey(pkey)
    req.sign(pkey, digest)
    return req