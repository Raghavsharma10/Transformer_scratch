def send_message(self, msg, stats=True):
        """ Send or queue outgoing message

        @param msg: Message to send

        @param stats: If set to True, will update statistics after operation
                      completes
        """
        self.send_jsonified(proto.json_encode(msg), stats)