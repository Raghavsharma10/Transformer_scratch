def uploadfile(baseurl, filename, format_, token, nonce, cert, method=requests.post):
    """Uploads file (given by `filename`) to server at `baseurl`.

    `sesson_key` and `nonce` are string values that get passed as POST
    parameters.
    """
    filehash = sha1sum(filename)
    files = {'filedata': open(filename, 'rb')}

    payload = {
        'sha1': filehash,
        'filename': os.path.basename(filename),
        'token': token,
        'nonce': nonce,
    }

    return method("%s/sign/%s" % (baseurl, format_), files=files, data=payload, verify=cert)