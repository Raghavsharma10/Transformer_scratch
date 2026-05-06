def create_bare(self):
        """
        Create instances for the Bare provider
        """
        self.instances = []
        for ip in self.settings['NODES']:
            new_instance = Instance.new(settings=self.settings, cluster=self)
            new_instance.ip = ip
            self.instances.append(new_instance)