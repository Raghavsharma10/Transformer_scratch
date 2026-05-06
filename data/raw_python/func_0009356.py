def make_sshable(c):
    """
    Set up passwordless SSH keypair & authorized_hosts access to localhost.
    """
    user = c.travis.sudo.user
    home = "~{0}".format(user)
    # Run sudo() as the new sudo user; means less chown'ing, etc.
    c.config.sudo.user = user
    ssh_dir = "{0}/.ssh".format(home)
    # TODO: worth wrapping in 'sh -c' and using '&&' instead of doing this?
    for cmd in ("mkdir {0}", "chmod 0700 {0}"):
        c.sudo(cmd.format(ssh_dir, user))
    c.sudo('ssh-keygen -f {0}/id_rsa -N ""'.format(ssh_dir))
    c.sudo("cp {0}/{{id_rsa.pub,authorized_keys}}".format(ssh_dir))