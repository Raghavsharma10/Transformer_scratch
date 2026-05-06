def get_zone_info(self, controller, zone, return_variable):
        """ Get all relevant info for the zone
            When called with return_variable == 4, then the function returns a list with current
             volume, source and ON/OFF status.
            When called with 0, 1 or 2, it will return an integer with the Power, Source and Volume """

        # Define the signature for a response message, used later to find the correct response from the controller.
        # FF is the hex we use to signify bytes that need to be ignored when comparing to response message.
        # resp_msg_signature = self.create_response_signature("04 02 00 @zz 07 00 00 01 00 0C", zone)

        _LOGGER.debug("Begin - controller= %s, zone= %s, get status", controller, zone)
        resp_msg_signature = self.create_response_signature("04 02 00 @zz 07", zone)
        send_msg = self.create_send_message("F0 @cc 00 7F 00 00 @kk 01 04 02 00 @zz 07 00 00", controller, zone)
        try:
            self.lock.acquire()
            _LOGGER.debug('Acquired lock for zone %s', zone)
            self.send_data(send_msg)
            _LOGGER.debug("Zone: %s Sent: %s", zone, send_msg)
            # Expected response is as per pg 23 of cav6.6_rnet_protocol_v1.01.00.pdf
            matching_message = self.get_response_message(resp_msg_signature)
            if matching_message is not None:
                # Offset of 11 is the position of return data payload is that we require for the signature we are using.
                _LOGGER.debug("matching message to use= %s", matching_message)
                _LOGGER.debug("matching message length= %s", len(matching_message))
                if return_variable == 4:
                    return_value = [matching_message[11], matching_message[12], matching_message[13]]
                else:
                    return_value = matching_message[return_variable + 11]
            else:
                return_value = None
                _LOGGER.warning("Did not receive expected Russound power state for controller %s and zone %s.", controller, zone)
        finally:
            self.lock.release()
            _LOGGER.debug("Released lock for zone %s", zone)
            _LOGGER.debug("End - controller= %s, zone= %s, get status \n", controller, zone)
            return return_value