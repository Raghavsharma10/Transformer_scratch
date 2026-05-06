def make_sftp(self):
        """Make SFTP client from open transport"""
        transport = self.client.get_transport()
        transport.open_session()
        return paramiko.SFTPClient.from_transport(transport)