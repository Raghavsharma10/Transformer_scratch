def reserve(self, force=False):
        """ Reserve port.

        XenaManager-2G -> Reserve/Relinquish Port.

        :param force: True - take forcefully, False - fail if port is reserved by other user
        """

        p_reservation = self.get_attribute('p_reservation')
        if p_reservation == 'RESERVED_BY_YOU':
            return
        elif p_reservation == 'RESERVED_BY_OTHER' and not force:
            raise TgnError('Port {} reserved by {}'.format(self, self.get_attribute('p_reservedby')))
        self.relinquish()
        self.send_command('p_reservation', 'reserve')