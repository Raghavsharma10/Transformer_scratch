def _send(self, msg):
        """
        Raw send to the given connection ID at the given uuid, mostly used 
        internally.
        """
        uuid = self.m2req.sender
        conn_id = self.m2req.conn_id

        header = "%s %d:%s," % (uuid, len(str(conn_id)), str(conn_id))
        zmq_message = header + ' ' + msg
        self.stream.send(zmq_message)