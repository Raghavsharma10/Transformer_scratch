def enable_node(self, service_name, node_name):
        """
        Enables a given node name for the given service name via the
        "enable server" HAProxy command.
        """
        logger.info("Enabling server %s/%s", service_name, node_name)
        return self.send_command(
            "enable server %s/%s" % (service_name, node_name)
        )