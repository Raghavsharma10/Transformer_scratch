def _authenticate_https(org):
    """Try authenticating via HTTPS, if succesful yields User, otherwise raises Error."""
    _CREDENTIAL_SOCKET.parent.mkdir(mode=0o700, exist_ok=True)
    try:
        Git.cache = f"-c credential.helper= -c credential.helper='cache --socket {_CREDENTIAL_SOCKET}'"
        git = Git(Git.cache)

        # Get credentials from cache if possible
        with _spawn(git("credential fill"), quiet=True) as child:
            child.sendline("protocol=https")
            child.sendline("host=github.com")
            child.sendline("")
            i = child.expect(["Username for '.+'", "Password for '.+'",
                              "username=([^\r]+)\r\npassword=([^\r]+)\r\n"])
            if i == 2:
                username, password = child.match.groups()
            else:
                username = password = None
                child.close()
                child.exitstatus = 0

        # No credentials found, need to ask user
        if password is None:
            username = _prompt_username(_("GitHub username: "))
            password = _prompt_password(_("GitHub password: "))

        # Check if credentials are correct
        res = requests.get("https://api.github.com/user", auth=(username, password))

        # Check for 2-factor authentication https://developer.github.com/v3/auth/#working-with-two-factor-authentication
        if "X-GitHub-OTP" in res.headers:
            raise Error("Looks like you have two-factor authentication enabled!"
                        " Please generate a personal access token and use it as your password."
                        " See https://help.github.com/articles/creating-a-personal-access-token-for-the-command-line for more info.")

        if res.status_code != 200:
            logger.info(res.headers)
            logger.info(res.text)
            raise Error(_("Invalid username and/or password.") if res.status_code ==
                        401 else _("Could not authenticate user."))

        # Canonicalize (capitalization of) username,
        # Especially if user logged in via email address
        username = res.json()["login"]

        # Credentials are correct, best cache them
        with _spawn(git("-c credentialcache.ignoresighup=true credential approve"), quiet=True) as child:
            child.sendline("protocol=https")
            child.sendline("host=github.com")
            child.sendline(f"path={org}/{username}")
            child.sendline(f"username={username}")
            child.sendline(f"password={password}")
            child.sendline("")

        yield User(name=username,
                   repo=f"https://{username}@github.com/{org}/{username}")
    except BaseException:
        # Some error occured while this context manager is active, best forget credentials.
        logout()
        raise