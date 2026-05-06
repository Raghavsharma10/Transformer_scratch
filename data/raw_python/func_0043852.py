def set_source(self, controller, zone, source):
        """ Set source for a zone - 0 based value for source """

        _LOGGER.info("Begin - controller= %s, zone= %s change source to %s.", controller, zone, source)
        send_msg = self.create_send_message("F0 @cc 00 7F 00 @zz @kk 05 02 00 00 00 F1 3E 00 00 00 @pr 00 01",
                                            controller, zone, source)
        try:
            self.lock.acquire()
            _LOGGER.debug('Zone %s - acquired lock for ', zone)
            self.send_data(send_msg)
            _LOGGER.debug("Zone %s - sent message %s", zone, send_msg)
            # Clear response buffer in case there is any response data(ensures correct results on future reads)
            self.get_response_message()
        finally:
            self.lock.release()
            _LOGGER.debug("Zone %s - released lock for ", zone)
            _LOGGER.debug("End - controller= %s, zone= %s source set to %s.\n", controller, zone, source)