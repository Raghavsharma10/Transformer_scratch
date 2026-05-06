def get_physical_port(self):
        """Returns the link aggregation object or the ethernet port object."""
        obj = None
        if self.is_link_aggregation():
            obj = UnityLinkAggregation.get(self._cli, self.get_id())
        else:
            obj = UnityEthernetPort.get(self._cli, self.get_id())
        return obj