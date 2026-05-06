def cli_info(self, event):
        """Provides information about the running instance"""

        self.log('Instance:', self.instance,
                 'Dev:', self.development,
                 'Host:', self.host,
                 'Port:', self.port,
                 'Insecure:', self.insecure,
                 'Frontend:', self.frontendtarget)