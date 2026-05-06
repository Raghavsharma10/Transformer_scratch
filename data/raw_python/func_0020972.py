def react(self, msg):
        """Called to provide a response to a message if needed.

        msg:
            This is a dictionary as returned by unpack_frame(...)
            or it can be a straight STOMP message. This function
            will attempt to determine which an deal with it.

        returned:
            A message to return or an empty string.

        """
        returned = ""

        # If its not a string assume its a dict.
        mtype = type(msg)
        if mtype in stringTypes:
            msg = unpack_frame(msg)
        elif mtype == dict:
            pass
        else:
            raise FrameError("Unknown message type '%s', I don't know what to do with this!" % mtype)

        if msg['cmd'] in self.states:
#            print("reacting to message - %s" % msg['cmd'])
            returned = self.states[msg['cmd']](msg)

        return returned