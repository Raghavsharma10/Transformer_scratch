def read_instances(self):
        """
        Read `.dsb/instances.yaml`
        Populates `self.cluster`
        """
        logger.debug('Reading instances from: %s', self.instances_path)
        if os.path.exists(self.instances_path):
            with open(self.instances_path, 'r') as f:
                list_ = yaml.load(f.read())
                self.cluster = Cluster.from_list(list_, settings=self.settings)