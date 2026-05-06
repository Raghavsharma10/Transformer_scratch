def toggle_mute(self, controller, zone):
        """ Toggle mute on/off for a zone
        Note: Not tested (acambitsis) """

        send_msg = self.create_send_message("F0 @cc 00 7F 00 @zz @kk 05 02 02 00 00 F1 40 00 00 00 0D 00 01",
                                            controller, zone)
        self.send_data(send_msg)
        self.get_response_message()