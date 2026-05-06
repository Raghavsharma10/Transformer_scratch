def _on_ws_message(self, ws, message):
        """
        on_message callback of websocket class, load the message into a dict and then
        update an Ack Object with the results
        :param ws: web socket connection that the message was received on
        :param message: web socket message in text form
        :return: None
        """
        logging.debug(message)
        json_list = json.loads(message)
        for rx_ack in json_list:
            ack = EventHub_pb2.Ack()
            for key, value in rx_ack.items():
                setattr(ack, key, value)
            self._publisher_callback(ack)