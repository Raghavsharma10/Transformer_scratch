def create_cloud(self):
        """
        Create instances for the cloud providers
        """
        instances = []
        for i in range(self.settings['NUMBER_NODES']):
            new_instance = Instance.new(settings=self.settings, cluster=self)
            instances.append(new_instance)

        create_nodes = [instance.create(suffix=i) for i, instance in enumerate(instances)]
        fetch_nodes = [instance.node for instance in instances]
        self.driver.wait_until_running(fetch_nodes)

        node_ids = [node.id for node in fetch_nodes]
        all_nodes = self.driver.list_nodes()
        new_nodes = [node for node in all_nodes if node.id in node_ids]
        for instance, node in zip(instances, new_nodes):
            instance.node = node

        self.instances = instances