def set_volume(self, controller, zone, volume):
        """ Set volume for zone to specific value.
        Divide the volume by 2 to translate to a range (0..50) as expected by Russound (Even thought the
        keypads show 0..100).
        """

        _LOGGER.debug("Begin - controller= %s, zone= %s, change volume to %s",controller, zone, volume)
        send_msg = self.create_send_message("F0 @cc 00 7F 00 00 @kk 05 02 02 00 00 F1 21 00 @pr 00 @zz 00 01",
                                            controller, zone, volume // 2)
        try:
            self.lock.acquire()
            _LOGGER.debug('Zone %s - acquired lock for ', zone)
            self.send_data(send_msg)
            _LOGGER.debug("Zone %s - sent message %s", zone, send_msg)
            self.get_response_message()  # Clear response buffer
        finally:
            self.lock.release()
            _LOGGER.debug("Zone %s - released lock for ", zone)
            _LOGGER.debug("End - controller %s, zone %s, volume set to %s.\n", controller, zone, volume)