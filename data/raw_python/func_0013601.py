def _decode_header(auth_header, client_id, client_secret):
    """
    Takes the header and tries to return an active token and decoded
    payload.
    :param auth_header:
    :param client_id:
    :param client_secret:
    :return: (token, profile)
    """
    try:
        token = auth_header.split()[1]
        payload = jwt.decode(
            token,
            client_secret,
            audience=client_id)
    except jwt.ExpiredSignature:
        raise exceptions.NotAuthorizedException(
            'Token has expired, please log in again.')
    # is valid client
    except jwt.InvalidAudienceError:
        message = 'Incorrect audience, expected: {}'.format(
            client_id)
        raise exceptions.NotAuthorizedException(message)
    # is valid token
    except jwt.DecodeError:
        raise exceptions.NotAuthorizedException(
            'Token signature could not be validated.')
    except Exception as e:
        raise exceptions.NotAuthorizedException(
            'Token signature was malformed. {}'.format(e.message))
    return token, payload