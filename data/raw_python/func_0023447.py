def from_raw(conn, raw):
        '''Return a new response from a raw buffer'''
        frame_type = struct.unpack('>l', raw[0:4])[0]
        message = raw[4:]
        if frame_type == FRAME_TYPE_MESSAGE:
            return Message(conn, frame_type, message)
        elif frame_type == FRAME_TYPE_RESPONSE:
            return Response(conn, frame_type, message)
        elif frame_type == FRAME_TYPE_ERROR:
            return Error(conn, frame_type, message)
        else:
            raise TypeError('Unknown frame type: %s' % frame_type)