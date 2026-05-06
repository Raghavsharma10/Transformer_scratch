def subscribe(self, tag, fun, description=None):
        """ Subscribe to something and register a function """
        self.methods[tag] = fun
        self.descriptions[tag] = description
        self.socket.set_string_option(nanomsg.SUB, nanomsg.SUB_SUBSCRIBE, tag)