def set_phy_mode(self, mode=IxePhyMode.ignore):
        """ Set phy mode to copper or fiber.
        :param mode: requested PHY mode.
        """
        if isinstance(mode, IxePhyMode):
            if mode.value:
                self.api.call_rc('port setPhyMode {} {}'.format(mode.value, self.uri))
        else:
            self.api.call_rc('port setPhyMode {} {}'.format(mode, self.uri))