def validate_config(cls, config):
        """
        Validates a config dictionary parsed from a cluster config file.

        Checks that a discovery method is defined and that at least one of
        the balancers in the config are installed and available.
        """
        if "discovery" not in config:
            raise ValueError("No discovery method defined.")

        installed_balancers = Balancer.get_installed_classes().keys()

        if not any([balancer in config for balancer in installed_balancers]):
            raise ValueError("No available balancer configs defined.")