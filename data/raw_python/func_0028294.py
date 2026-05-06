def up(self):
        """
                Function operates on the IMCInterface object and configures the interface into an
                administratively up state and refreshes contents of self.adminstatus
                :return:
                """
        set_interface_up(self.ifIndex, self.auth, self.url, devip=self.ip)
        self.adminstatus = get_interface_details(self.ifIndex, self.auth, self.url, devip=self.ip)[
            'adminStatusDesc']