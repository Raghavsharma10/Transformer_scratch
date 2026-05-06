def _ssh_client(self):
        """Gets an SSH client to connect with.
        """
        ssh = paramiko.SSHClient()
        ssh.load_system_host_keys()
        ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
        return ssh