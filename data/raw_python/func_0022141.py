def jdout(api_response):
    """
    JD Output function. Does quick pretty printing of a CloudGenix Response body. This function returns a string
    instead of directly printing content.

      **Parameters:**

      - **api_response:** A CloudGenix-attribute extended `requests.Response` object

    **Returns:** Pretty-formatted text of the Response body
    """
    try:
        # attempt to output the cgx_content. should always be a Dict if it exists.
        output = json.dumps(api_response.cgx_content, indent=4)
    except (TypeError, ValueError, AttributeError):
        # cgx_content did not exist, or was not JSON serializable. Try pretty output the base obj.
        try:
            output = json.dumps(api_response, indent=4)
        except (TypeError, ValueError, AttributeError):
            # Same issue, just raw output the passed data. Let any exceptions happen here.
            output = api_response
    return output