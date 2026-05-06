def get_identity(identity):
    """Returns some information about the currently authenticated identity"""
    return flask.Response(
        json.dumps(
            {
                'identity': {
                    'id': identity.id,
                    'etag': identity.etag,
                    'name': identity.name,
                    'fullname': identity.fullname,
                    'email': identity.email,
                    'timezone': identity.timezone,
                    'teams': _encode_dict(identity.teams)
                }
            }
        ), 200,
        headers={'ETag': identity.etag},
        content_type='application/json'
    )