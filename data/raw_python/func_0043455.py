def put(self, local, remote, sudo=False):
        """Copy local file to host via SFTP/SCP

        Copy is done natively using SFTP/SCP version 2 protocol, no scp command
        is used or required.
        """
        if(os.path.isdir(local)):
            self.put_dir(local, remote, sudo=sudo)
        else:
            self.put_single(local, remote, sudo=sudo)