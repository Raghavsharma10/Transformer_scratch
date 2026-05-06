def install_plugin(username, repo):
    """Installs a Blended plugin from GitHub"""
    print("Installing plugin from " + username + "/" + repo)

    pip.main(['install', '-U', "git+git://github.com/" +
              username + "/" + repo + ".git"])