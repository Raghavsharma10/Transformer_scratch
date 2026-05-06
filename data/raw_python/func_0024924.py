def _send_to_timeseries(self, message):
        """
        Establish or reuse socket connection and send
        the given message to the timeseries service.
        """
        logging.debug("MESSAGE=" + str(message))

        result = None
        try:
            ws = self._get_websocket()
            ws.send(json.dumps(message))
            result = ws.recv()
        except (websocket.WebSocketConnectionClosedException, Exception) as e:
            logging.debug("Connection failed, will try again.")
            logging.debug(e)

            ws = self._get_websocket(reuse=False)
            ws.send(json.dumps(message))
            result = ws.recv()

        logging.debug("RESULT=" + str(result))
        return result