def handle_stream(self, message):
        """
        Acts on message reception
        :param message: string of an incoming message

        parse all the fields and builds an Event object that is passed to the callback function
        """
        logging.debug("handle_stream(...)")

        event = Event()
        for line in message.strip().splitlines():
            (field, value) = line.split(":", 1)
            field = field.strip()

            if field == "event":
                event.name = value.lstrip()
            elif field == "data":
                value = value.lstrip()
                if event.data is None:
                    event.data = value
                else:
                    event.data = "%s\n%s" % (event.data, value)
            elif field == "id":
                event.id = value.lstrip()
                self.last_event_id = event.id
            elif field == "retry":
                try:
                    self.retry_timeout = int(value)
                    event.retry = self.retry_timeout
                    logging.info("timeout reset: %s" % (value,))
                except ValueError:
                    pass
            elif field == "":
                logging.debug("received comment: %s" % (value,))
            else:
                raise Exception("Unknown field !")
        self.events.append(event)