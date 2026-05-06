def json_success_response(data, response):
    """
    Formats the response of a successful token request as JSON.

    Also adds default headers and status code.
    """
    response.body = json.dumps(data)
    response.status_code = 200

    response.add_header("Content-Type", "application/json")
    response.add_header("Cache-Control", "no-store")
    response.add_header("Pragma", "no-cache")