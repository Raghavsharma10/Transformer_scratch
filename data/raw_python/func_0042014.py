def __on_presence(self, data):
        """
        Got a presence stanza
        """
        room_jid = data['from'].bare
        muc_presence = data['muc']
        room = muc_presence['room']
        nick = muc_presence['nick']

        with self.__lock:
            try:
                # Get room state machine
                room_data = self.__rooms[room]
                if room_data.nick != nick:
                    # Not about the room creator
                    return
            except KeyError:
                # Unknown room (or not a room)
                return
            else:
                # Clean up, as we got what we wanted
                del self.__rooms[room]

            if not self.__rooms:
                # No more rooms: no need to listen to presence anymore
                self.__xmpp.del_event_handler("presence", self.__on_presence)

        if data['type'] == 'error':
            # Got an error: update the state machine and clean up
            self.__safe_errback(room_data, data['error']['condition'],
                                data['error']['text'])

        elif muc_presence['affiliation'] != 'owner':
            # We are not the owner the room: consider it an error
            self.__safe_errback(room_data, 'not-owner',
                                'We are not the owner of the room')

        else:
            # Success: we own the room
            # Setup room configuration
            try:
                config = self.__muc.getRoomConfig(room_jid)
            except ValueError:
                # Can't differentiate IQ errors from a "no configuration"
                # result: consider it OK
                self.__logger.warning("Can't get the configuration form for "
                                      "XMPP room %s", room_jid)
                self.__safe_callback(room_data)
            else:
                # Prepare our configuration
                custom_values = room_data.configuration or {}

                # Filter options that are not known from the server
                known_fields = config['fields']
                to_remove = [key for key in custom_values
                             if key not in known_fields]
                for key in to_remove:
                    del custom_values[key]

                # Send configuration (use a new form to avoid OpenFire to have
                # an internal error)
                form = self.__xmpp['xep_0004'].make_form("submit")
                form['values'] = custom_values
                self.__muc.setRoomConfig(room_jid, form)

                # Call back the creator
                self.__safe_callback(room_data)