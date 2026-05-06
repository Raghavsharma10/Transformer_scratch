def http_request_headers(instance):
    """Ensure the keys of the 'request_headers' property of the http-request-
    ext extension of network-traffic objects conform to the format for HTTP
    request headers. Use a regex because there isn't a definitive source.
    https://www.iana.org/assignments/message-headers/message-headers.xhtml does
    not differentiate between request and response headers, and leaves out
    several common non-standard request fields listed elsewhere.
    """
    for key, obj in instance['objects'].items():
        if ('type' in obj and obj['type'] == 'network-traffic'):
            try:
                headers = obj['extensions']['http-request-ext']['request_header']
            except KeyError:
                continue

            for hdr in headers:
                if hdr not in enums.HTTP_REQUEST_HEADERS:
                    yield JSONError("The 'request_header' property of object "
                                    "'%s' contains an invalid HTTP request "
                                    "header ('%s')."
                                    % (key, hdr), instance['id'],
                                    'http-request-headers')