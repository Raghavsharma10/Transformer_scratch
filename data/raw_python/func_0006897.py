def _responsify(api_spec, error, status):
    """Take a bravado-core model representing an error, and return a Flask Response
    with the given error code and error instance as body"""
    result_json = api_spec.model_to_json(error)
    r = jsonify(result_json)
    r.status_code = status
    return r