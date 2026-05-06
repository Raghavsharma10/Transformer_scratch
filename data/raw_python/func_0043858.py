def create_response_signature(self, string_message, zone):
        """ Basic helper function to keep code clean for defining a response message signature """

        zz = ''
        if zone is not None:
            zz = hex(int(zone)-1).replace('0x', '')  # RNET requires zone value to be zero based
        string_message = string_message.replace('@zz', zz)  # Replace zone parameter
        return string_message