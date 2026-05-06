def from_list(cls, values, settings):
        """
        From a list of dicts (each dict is an `Instance.to_dict`)
        """
        logger.debug('Creating Cluster from list')
        self = cls()
        self.settings = settings
        self.instances = []

        for instance in values:
            uid, ip, port = instance['uid'], instance['ip'], instance['port']
            new_instance = Instance.new(settings=settings, cluster=self, uid=uid, ip=ip, port=port)
            self.instances.append(new_instance)
        return self