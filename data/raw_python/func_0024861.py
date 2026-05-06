def _init_publisher_ws(self):
        """
        Create a new web socket connection with proper headers.
        """
        logging.debug("Initializing new web socket connection.")

        url = ('wss://%s/v1/stream/messages/' % self.eventhub_client.host)

        headers = self._generate_publish_headers()

        logging.debug("URL=" + str(url))
        logging.debug("HEADERS=" + str(headers))

        websocket.enableTrace(False)
        self._ws = websocket.WebSocketApp(url,
                                          header=headers,
                                          on_message=self._on_ws_message,
                                          on_open=self._on_ws_open,
                                          on_close=self._on_ws_close)
        self._ws_thread = threading.Thread(target=self._ws.run_forever, kwargs={'ping_interval': 30})
        self._ws_thread.daemon = True
        self._ws_thread.start()
        time.sleep(1)