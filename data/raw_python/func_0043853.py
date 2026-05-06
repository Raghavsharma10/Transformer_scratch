def all_on_off(self, power):
        """ Turn all zones on or off
        Note that the all on function is not supported by the Russound CAA66, although it does support the all off.
        On and off are supported by the CAV6.6.
        Note: Not tested (acambitsis)
        """

        send_msg = self.create_send_message("F0 7F 00 7F 00 00 @kk 05 02 02 00 00 F1 22 00 00 @pr 00 00 01",
                                            None, None, power)
        self.send_data(send_msg)
        self.get_response_message()