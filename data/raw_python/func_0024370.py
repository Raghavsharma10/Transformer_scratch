def validate_request_certificate(headers, data):
    """Ensure that the certificate and signature specified in the
    request headers are truely from Amazon and correctly verify.

    Returns True if certificate verification succeeds, False otherwise.

    :param headers: Dictionary (or sufficiently dictionary-like) map of request
        headers.
    :param data: Raw POST data attached to this request.
    """

    # Make sure we have the appropriate headers.
    if 'SignatureCertChainUrl' not in headers or \
       'Signature' not in headers:
        log.error('invalid request headers')
        return False

    cert_url = headers['SignatureCertChainUrl']
    sig = base64.b64decode(headers['Signature'])

    cert = _get_certificate(cert_url)

    if not cert:
        return False

    try:
        # ... wtf kind of API decision is this
        crypto.verify(cert, sig, data, 'sha1')
        return True
    except:
        log.error('invalid request signature')
        return False