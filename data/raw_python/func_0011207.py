def _repo_is_preset(repo):
    """Evaluate whether GitHub repository is a be package

    Arguments:
        gist (str): username/id pair e.g. mottosso/be-ad

    """

    package_template = "https://raw.githubusercontent.com"
    package_template += "/{repo}/master/package.json"
    package_path = package_template.format(repo=repo)

    response = get(package_path)
    if response.status_code == 404:
        return False

    try:
        data = response.json()
    except:
        return False

    if not data.get("type") == "bepreset":
        return False

    return True