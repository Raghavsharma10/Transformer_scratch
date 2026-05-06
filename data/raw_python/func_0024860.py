def _publish_queue_wss(self):
        """
        send the messages down the web socket connection as a json object
        :return: None
        """

        msg = []
        for m in self._tx_queue:
            msg.append({'id': m.id, 'body': m.body, 'zone_id': m.zone_id})
        self._ws.send(json.dumps(msg), opcode=websocket.ABNF.OPCODE_BINARY)