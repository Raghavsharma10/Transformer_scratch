def json_error_response(error, response, status_code=400):
    """
    Formats an error as a response containing a JSON body.
    """
    msg = {"error": error.error, "error_description": error.explanation}

    response.status_code = status_code
    response.add_header("Content-Type", "application/json")
    response.body = json.dumps(msg)

    return response