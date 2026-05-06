def info(self):
        """
        Description:

            Returns dict of information about the TV
            name, model, version

        """
        return {"name": self._send_command('name'),
                "model": self._send_command('model'),
                "version": self._send_command('version'),
                "ip_version": self._send_command('ip_version')
               }