def dataReceived(self, data):
        """Data received, react to it and respond if needed.
        """
#        print "receiver dataReceived: <%s>" % data
        
        msg = stomper.unpack_frame(data)
        
        returned = self.sm.react(msg)

#        print "receiver returned <%s>" % returned
        
        if returned:
            self.transport.write(returned)