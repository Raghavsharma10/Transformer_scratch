def _github_ssh_key_exists(cls):
        """Returns True if any key on Github matches a local key, else False."""
        remote_keys = map(lambda k: k._key, cls._user.get_keys())
        found = False
        pubkey_files = glob.glob(os.path.expanduser('~/.ssh/*.pub'))
        for rk in remote_keys:
            for pkf in pubkey_files:
                local_key = io.open(pkf, encoding='utf-8').read()
                # in PyGithub 1.23.0, remote key is an object, not string
                rkval = rk if isinstance(rk, six.string_types) else rk.value
                # don't use "==" because we have comments etc added in public_key
                if rkval in local_key:
                    found = True
                    break
        return found