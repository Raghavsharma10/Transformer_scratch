def verify_url_path(url_path, query_args, secret_key, salt_arg='_', max_expiry=None, digest=None):
    # type: (str, Dict[str, str], bytes, str, int, Callable) -> bool
    """
    Verify a URL path is correctly signed.

    :param url_path: URL path
    :param secret_key: Signing key
    :param query_args: Arguments that make up the query string
    :param salt_arg: Argument required for salt (set to None to disable)
    :param max_expiry: Maximum length of time an expiry value can be for (set to None to disable)
    :param digest: Specify the digest function to use; default is sha256 from hashlib
    :rtype: bool
    :raises: URLError

    """
    try:
        supplied_signature = query_args.pop('signature')
    except KeyError:
        raise SigningError("Signature missing.")

    if salt_arg is not None and salt_arg not in query_args:
        raise SigningError("No salt used.")

    if max_expiry is not None and 'expires' not in query_args:
        raise SigningError("Expiry time is required.")

    # Validate signature
    signature = _generate_signature(url_path, secret_key, query_args, digest)
    if not hmac.compare_digest(signature, supplied_signature):
        raise SigningError('Signature not valid.')

    # Check expiry
    try:
        expiry_time = int(query_args.pop('expires'))
    except KeyError:
        pass  # URL doesn't have an expire time
    except ValueError:
        raise SigningError("Invalid expiry value.")
    else:
        expiry_delta = expiry_time - time()
        if expiry_delta < 0:
            raise SigningError("Signature has expired.")
        if max_expiry and expiry_delta > max_expiry:
            raise SigningError("Expiry time out of range.")

    return True