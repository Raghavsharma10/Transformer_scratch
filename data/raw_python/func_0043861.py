def find_signature(self, data_stream, msg_signature):
        """ Takes the stream of bytes received and looks for a message that matches the signature
        of the expected response """

        signature_match_index = None  # The message that will be returned if it matches the signature
        msg_signature = msg_signature.split()  # Split into list
        # convert to bytearray in order to be able to compare with the messages list which contains bytearrays
        msg_signature = bytearray(int(x, 16) for x in msg_signature)
        # loop through each message returned from Russound
        index_of_last_f7 = None
        for i in range(len(data_stream)):
            if data_stream[i] == 247:
                index_of_last_f7 = i
            # the below line checks for the matching signature, ensuring ALL bytes of the response have been received
            if (data_stream[i:i + len(msg_signature)] == msg_signature) and (len(data_stream) - i >= 24):
                signature_match_index = i
                break
        if signature_match_index is None:
            # Scrap bytes up to end of msg (to avoid searching these again)
            data_stream = data_stream[index_of_last_f7:len(data_stream)]
            matching_message = None
        else:
            matching_message = data_stream[signature_match_index:len(data_stream)]

        _LOGGER.debug("Message signature found at location: %s", signature_match_index)
        return matching_message, data_stream