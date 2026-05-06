def salt(self, module, target='*', args=None, kwargs=None, ssh=False):
        """
        Execute a salt (or salt-ssh) command
        """
        if ssh:
            return salt.salt_ssh(self, target, module, args, kwargs)
        else:
            return salt.salt_master(self, target, module, args, kwargs)