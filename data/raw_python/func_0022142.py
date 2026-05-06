def jdout_detailed(api_response, sensitive=False):
    """
    JD Output Detailed function. Meant for quick DETAILED pretty-printing of CloudGenix Request and Response
    objects for troubleshooting. This function returns a string instead of directly printing content.

      **Parameters:**

      - **api_response:** A CloudGenix-attribute extended `requests.Response` object
      - **sensitive:** Boolean, if True will print sensitive content (specifically, authentication cookies/headers).

    **Returns:** Pretty-formatted text of the Request, Request Headers, Request body, Response, Response Headers,
    and Response Body.
    """
    try:
        # try to be super verbose.
        output = "REQUEST: {0} {1}\n".format(api_response.request.method, api_response.request.path_url)
        output += "REQUEST HEADERS:\n"
        for key, value in api_response.request.headers.items():
            # look for sensitive values
            if key.lower() in ['cookie'] and not sensitive:
                # we need to do some work to watch for the AUTH_TOKEN cookie. Split on cookie separator
                cookie_list = value.split('; ')
                muted_cookie_list = []
                for cookie in cookie_list:
                    # check if cookie starts with a permutation of AUTH_TOKEN/whitespace.
                    if cookie.lower().strip().startswith('auth_token='):
                        # first 11 chars of cookie with whitespace removed + mute string.
                        newcookie = cookie.strip()[:11] + "\"<SENSITIVE - NOT SHOWN BY DEFAULT>\""
                        muted_cookie_list.append(newcookie)
                    else:
                        muted_cookie_list.append(cookie)
                # got list of cookies, muted as needed. recombine.
                muted_value = "; ".join(muted_cookie_list)
                output += "\t{0}: {1}\n".format(key, muted_value)
            elif key.lower() in ['x-auth-token'] and not sensitive:
                output += "\t{0}: {1}\n".format(key, "<SENSITIVE - NOT SHOWN BY DEFAULT>")
            else:
                output += "\t{0}: {1}\n".format(key, value)
        # if body not present, output blank.
        if not api_response.request.body:
            output += "REQUEST BODY:\n{0}\n\n".format({})
        else:
            try:
                # Attempt to load JSON from string to make it look beter.
                output += "REQUEST BODY:\n{0}\n\n".format(json.dumps(json.loads(api_response.request.body), indent=4))
            except (TypeError, ValueError, AttributeError):
                # if pretty call above didn't work, just toss it to jdout to best effort it.
                output += "REQUEST BODY:\n{0}\n\n".format(jdout(api_response.request.body))
        output += "RESPONSE: {0} {1}\n".format(api_response.status_code, api_response.reason)
        output += "RESPONSE HEADERS:\n"
        for key, value in api_response.headers.items():
            output += "\t{0}: {1}\n".format(key, value)
        try:
            # look for CGX content first.
            output += "RESPONSE DATA:\n{0}".format(json.dumps(api_response.cgx_content, indent=4))
        except (TypeError, ValueError, AttributeError):
            # look for standard response data.
            output += "RESPONSE DATA:\n{0}".format(json.dumps(json.loads(api_response.content), indent=4))
    except (TypeError, ValueError, AttributeError, UnicodeDecodeError):
        # cgx_content did not exist, or was not JSON serializable. Try pretty output the base obj.
        try:
            output = json.dumps(api_response, indent=4)
        except (TypeError, ValueError, AttributeError):
            # Same issue, just raw output the passed data. Let any exceptions happen here.
            output = api_response
    return output