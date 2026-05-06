def apply_config(self, config):
        """
        Sets the `discovery` and `meta_cluster` attributes, as well as the
        configured + available balancer attributes from a given validated
        config.
        """
        self.discovery = config["discovery"]
        self.meta_cluster = config.get("meta_cluster")
        for balancer_name in Balancer.get_installed_classes().keys():
            if balancer_name in config:
                setattr(self, balancer_name, config[balancer_name])