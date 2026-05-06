def disable_node(self, service_name, node_name):
        """
        Disables a given node name for the given service name via the
        "disable server" HAProxy command.
        """
        logger.info("Disabling server %s/%s", service_name, node_name)
        return self.send_command(
            "disable server %s/%s" % (service_name, node_name)
        )