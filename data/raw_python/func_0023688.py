def reconnected(self, conn):
        '''Subscribe connection and manipulate its RDY state'''
        conn.sub(self._topic, self._channel)
        conn.rdy(1)