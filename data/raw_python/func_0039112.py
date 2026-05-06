def _authenticate_ssh(org):
    """Try authenticating via ssh, if succesful yields a User, otherwise raises Error."""
    # Try to get username from git config
    username = os.environ.get(f"{org.upper()}_USERNAME")
    # Require ssh-agent
    child = pexpect.spawn("ssh -T git@github.com", encoding="utf8")
    # GitHub prints 'Hi {username}!...' when attempting to get shell access
    i = child.expect(["Hi (.+)! You've successfully authenticated",
                      "Enter passphrase for key",
                      "Permission denied",
                      "Are you sure you want to continue connecting"])
    child.close()

    if i == 0:
        if username is None:
            username = child.match.groups()[0]
    else:
        return None

    return User(name=username,
                repo=f"git@github.com:{org}/{username}")