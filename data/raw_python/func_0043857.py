def create_send_message(self, string_message, controller, zone=None, parameter=None):
        """ Creates a message from a string, substituting the necessary parameters,
        that is ready to send to the socket """

        cc = hex(int(controller) - 1).replace('0x', '')  # RNET requires controller value to be zero based
        if zone is not None:
            zz = hex(int(zone) - 1).replace('0x', '')  # RNET requires zone value to be zero based
        else:
            zz = ''
        if parameter is not None:
            pr = hex(int(parameter)).replace('0x', '')
        else:
            pr = ''

        string_message = string_message.replace('@cc', cc)  # Replace controller parameter
        string_message = string_message.replace('@zz', zz)  # Replace zone parameter
        string_message = string_message.replace('@kk', KEYPAD_CODE)  # Replace keypad parameter
        string_message = string_message.replace('@pr', pr)  # Replace specific parameter to message

        # Split message into an array for each "byte" and add the checksum and end of message bytes
        send_msg = string_message.split()
        send_msg = self.calc_checksum(send_msg)
        return send_msg