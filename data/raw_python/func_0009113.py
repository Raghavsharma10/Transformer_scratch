def res_to_str(res):
    """
    :param res: :class:`requests.Response` object

    Parse the given request and generate an informative string from it
    """
    if 'Authorization' in res.request.headers:
        res.request.headers['Authorization'] = "*****"
    return """
####################################
url = %s
headers = %s
-------- data sent -----------------
%s
------------------------------------
@@@@@ response @@@@@@@@@@@@@@@@
headers = %s
code = %d
reason = %s
--------- data received ------------
%s
------------------------------------
####################################
""" % (res.url,
       str(res.request.headers),
       OLD_REQ and res.request.data or res.request.body,
       res.headers,
       res.status_code,
       res.reason,
       res.text)