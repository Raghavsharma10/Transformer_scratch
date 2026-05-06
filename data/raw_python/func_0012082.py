def run(self):
        """
        Called by internal API subsystem to initialize websockets connections
        in the API interface
        """
        self.api = self.context.get("cls")(self.context)
        self.context["inst"].append(self)  # Adapters used by strategies

        def on_ws_connect(*args, **kwargs):
            """Callback on connect hook to set is_connected_ws"""
            self.is_connected_ws = True
            self.api.on_ws_connect(*args, **kwargs)

        # Initialize websocket in a thread with channels
        if hasattr(self.api, "on_ws_connect"):
            self.thread = Process(target=self.api.connect_ws, args=(
                on_ws_connect, [
                    SockChannel(channel, res_type, self._generate_result)
                    for channel, res_type in
                    self
                    .context
                    .get("conf")
                    .get("subscriptions")
                    .items()
                ]))
            self.thread.start()