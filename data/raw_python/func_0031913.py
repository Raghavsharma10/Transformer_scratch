def install_template(username, repo):
    """Installs a Blended template from GitHub"""
    print("Installing template from " + username + "/" + repo)

    dpath = os.path.join(cwd, "templates")
    getunzipped(username, repo, dpath)