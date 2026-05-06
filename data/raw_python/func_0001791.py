def check(response, expected_status=200, url=None):
    """
    Check whether the status code of the response equals expected_status and
    raise an APIError otherwise.
    @param url: The url of the response (for error messages).
                Defaults to response.url
    @param json: if True, return r.json(), otherwise return r.text
    """
    if response.status_code != expected_status:
        if url is None:
            url = response.url

        try:
            err = response.json()
        except:
            err = {} # force generic error

        if all(x in err for x in ("status", "message", "description", "details")):
            raise _APIError(err["status"], err['message'], url,
                           err, err["description"], err["details"])
        else: # generic error
            suffix = ".html" if "<html" in response.text else ".txt"
            msg = response.text
            if len(msg) > 200:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                    f.write(response.text.encode("utf-8"))
                msg = "{}...\n\n[snipped; full response written to {f.name}".format(msg[:100], **locals())
                
            msg = ("Request {url!r} returned code {response.status_code},"
                   " expected {expected_status}. \n{msg}".format(**locals()))
            raise _APIError(response.status_code, msg, url, response.text)
    if response.headers.get('Content-Type') == 'application/json':
        try:
            return response.json()
        except:
            raise Exception("Cannot decode json; text={response.text!r}"
                            .format(**locals()))
    else:
        return response.text