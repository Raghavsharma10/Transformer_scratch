def set_power(self, controller, zone, power):
        """ Switch power on/off to a zone
        :param controller: Russound Controller ID. For systems with one controller this should be a value of 1.
        :param zone: The zone to be controlled. Expect a 1 based number.
        :param power: 0 = off, 1 = on
        """

        _LOGGER.debug("Begin - controller= %s, zone= %s, change power to %s",controller, zone, power)
        send_msg = self.create_send_message("F0 @cc 00 7F 00 00 @kk 05 02 02 00 00 F1 23 00 @pr 00 @zz 00 01",
                                            controller, zone, power)
        try:
            self.lock.acquire()
            _LOGGER.debug('Zone %s - acquired lock for ', zone)
            self.send_data(send_msg)
            _LOGGER.debug("Zone %s - sent message %s", zone, send_msg)
            self.get_response_message()  # Clear response buffer
        finally:
            self.lock.release()
            _LOGGER.debug("Zone %s - released lock for ", zone)
            _LOGGER.debug("End - controller %s, zone %s, power set to %s.\n", controller, zone, power)