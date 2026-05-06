def html_from_markdown(markdown):
    """ Takes raw markdown, returns html result from GitHub api """

    if login:
        r = requests.get(gh_url+"/rate_limit", auth=login.auth())
        if r.status_code >= 400:
            if r.status_code != 401:
                err = RequestError('Bad HTTP Status Code: %s' % r.status_code)
                raise err
            if verbose:
                sys.stderr.write('Unauthorized. Proceeding without login...\n')
            login.devalue()

    headers = {'content-type': 'text/plain', 'charset': 'utf-8'}

    r = requests.post(gh_url + "/markdown/raw", data=markdown.encode('utf-8'),
                      auth=login.auth(), headers=headers)
    if r.status_code >= 400 and r.status_code != 403:
            err = RequestError('Bad HTTP Status Code: %s' % r.status_code)
            raise err

    if verbose:
        sys.stderr.write("%s requests remaining, resets in %d minutes\n"
                         % rate_limit_info())
    return r.text