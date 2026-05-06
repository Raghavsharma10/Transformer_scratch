def check_response_code(resp):
    """
    check if query quota has been surpassed or other errors occured
    :param resp: json response
    :return:
    """
    if resp["status"] == "OK" or resp["status"] == "ZERO_RESULTS":
        return

    if resp["status"] == "REQUEST_DENIED":
        raise Exception("Google Places " + resp["status"],
                        "Request was denied, the API key is invalid.")

    if resp["status"] == "OVER_QUERY_LIMIT":
        raise Exception("Google Places " + resp["status"],
                        "You exceeded your Query Limit for Google Places API Web Service, "
                        "check https://developers.google.com/places/web-service/usage "
                        "to upgrade your quota.")

    if resp["status"] == "INVALID_REQUEST":
        raise Exception("Google Places " + resp["status"],
                        "The query string is malformed, "
                        "check if your formatting for lat/lng and radius is correct.")

    raise Exception("Google Places " + resp["status"],
                    "Unidentified error with the Places API, please check the response code")