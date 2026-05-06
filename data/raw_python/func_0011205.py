def get(path, **kwargs):
    """requests.get wrapper"""
    token = os.environ.get(BE_GITHUB_API_TOKEN)
    if token:
        kwargs["headers"] = {
            "Authorization": "token %s" % token
        }

    try:
        response = requests.get(path, verify=False, **kwargs)
        if response.status_code == 403:
            lib.echo("Patience: You can't pull more than 60 "
                     "presets per hour without an API token.\n"
                     "See https://github.com/mottosso/be/wiki"
                     "/advanced#extended-preset-access")
            sys.exit(lib.USER_ERROR)
        return response
    except Exception as e:
        if self.verbose:
            lib.echo("ERROR: %s" % e)
        else:
            lib.echo("ERROR: Something went wrong. "
                     "See --verbose for more information")