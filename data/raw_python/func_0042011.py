def create_room(self, room, service, nick, config=None,
                    callback=None, errback=None, room_jid=None):
        """
        Prepares the creation of a room.

        The callback is a method with two arguments:
          - room: Bare JID of the room
          - nick: Nick used to create the room

        The errback is a method with 4 arguments:
          - room: Bare JID of the room
          - nick: Nick used to create the room
          - condition: error category (XMPP specification or "not-owner")
          - text: description of the error


        :param room: Name of the room
        :param service: Name of the XMPP MUC service
        :param config: Configuration of the room
        :param callback: Method called back on success
        :param errback: Method called on error
        :param room_jid: Forced room JID
        """
        self.__logger.debug("Creating room: %s", room)

        with self.__lock:
            if not room_jid:
                # Generate/Format the room JID if not given
                room_jid = sleekxmpp.JID(local=room, domain=service).bare

            self.__logger.debug("... Room JID: %s", room_jid)

            if not self.__rooms:
                # First room to create: register to events
                self.__xmpp.add_event_handler("presence", self.__on_presence)

            # Store information
            self.__rooms[room_jid] = RoomData(room_jid, nick, config,
                                              callback, errback)

        # Send the presence, i.e. request creation of the room
        self.__muc.joinMUC(room_jid, nick)