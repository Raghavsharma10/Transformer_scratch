def _gist_is_preset(repo):
    """Evaluate whether gist is a be package

    Arguments:
        gist (str): username/id pair e.g. mottosso/2bb4651a05af85711cde

    """

    _, gistid = repo.split("/")

    gist_template = "https://api.github.com/gists/{}"
    gist_path = gist_template.format(gistid)

    response = get(gist_path)
    if response.status_code == 404:
        return False

    try:
        data = response.json()
    except:
        return False

    files = data.get("files", {})
    package = files.get("package.json", {})

    try:
        content = json.loads(package.get("content", ""))
    except:
        return False

    if content.get("type") != "bepreset":
        return False

    return True