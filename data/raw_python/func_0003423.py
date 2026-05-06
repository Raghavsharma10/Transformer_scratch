def decode_cert(cert):
    """Convert an X509 certificate into a Python dictionary

    This function converts the given X509 certificate into a Python dictionary
    in the manner established by the Python standard library's ssl module.
    """

    ret_dict = {}
    subject_xname = X509_get_subject_name(cert.value)
    ret_dict["subject"] = _create_tuple_for_X509_NAME(subject_xname)

    notAfter = X509_get_notAfter(cert.value)
    ret_dict["notAfter"] = ASN1_TIME_print(notAfter)

    peer_alt_names = _get_peer_alt_names(cert)
    if peer_alt_names is not None:
        ret_dict["subjectAltName"] = peer_alt_names

    return ret_dict