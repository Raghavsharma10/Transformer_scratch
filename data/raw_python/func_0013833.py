def get_file_port(self):
        """Returns ports list can be used by File

        File ports includes ethernet ports and link aggregation ports.
        """
        eths = self.get_ethernet_port(bond=False)
        las = self.get_link_aggregation()
        return eths + las