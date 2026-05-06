def override_ssh_auth_env():
    """Override the `$SSH_AUTH_SOCK `env variable to mock the absence of an SSH agent."""
    ssh_auth_sock = "SSH_AUTH_SOCK"
    old_ssh_auth_sock = os.environ.get(ssh_auth_sock)

    del os.environ[ssh_auth_sock]

    yield

    if old_ssh_auth_sock:
        os.environ[ssh_auth_sock] = old_ssh_auth_sock