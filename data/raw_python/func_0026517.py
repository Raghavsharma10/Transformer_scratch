def clientdisconnect(self, event):
        """Handler to deal with a possibly disconnected remote controlling
        client
        :param event: ClientDisconnect Event
        """

        try:
            if event.clientuuid == self.remote_controller:
                self.log("Remote controller disconnected!", lvl=critical)
                self.remote_controller = None
        except Exception as e:
            self.log("Strange thing while client disconnected", e, type(e))