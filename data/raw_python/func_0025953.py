def pack(self):
        """Return binary format of packet.

           The returned string is the binary format of the packet with
           stuffing and framing applied. It is ready to be sent to
           the GPS.

        """

        # Possible structs for packet ID.
        #
        try:
            structs_ = get_structs_for_fields([self.fields[0]])
        except (TypeError):
            # TypeError, if self.fields[0] is a wrong argument to `chr()`.
            raise PackError(self)


        # Possible structs for packet ID + subcode
        #
        if structs_ == []:
            try:
                structs_ = get_structs_for_fields([self.fields[0], self.fields[1]])
            except (IndexError, TypeError):
                # IndexError, if no self.fields[1]
                # TypeError, if self.fields[1] is a wrong argument to `chr()`.
                raise PackError(self)


        # Try to pack the packet with any of the possible structs.
        #
        for struct_ in structs_:
            try:
                return struct_.pack(*self.fields)
            except struct.error:
                pass

        # We only get here if the ``return`` inside the``for`` loop
        # above wasn't reached, i.e. none of the `structs_` matched.
        #
        raise PackError(self)