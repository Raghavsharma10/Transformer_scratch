def ssh_session(key_filename,
                username,
                ip_address,
                *cli):
    """ opens a ssh shell to the host """
    local('ssh -t -i %s %s@%s %s' % (key_filename,
                                     username,
                                     ip_address,
                                     "".join(chain.from_iterable(cli))))