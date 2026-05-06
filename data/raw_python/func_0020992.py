def dataReceived(self, data):
        """Use stompbuffer to determine when a complete message has been received. 
        """
        self.stompBuffer.appendData(data)

        while True:
           msg = self.stompBuffer.getOneMessage()
           if msg is None:
               break

           returned = self.react(msg)
           if returned:
               self.transport.write(returned)