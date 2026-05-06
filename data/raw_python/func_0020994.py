def ack(self, msg):
        """Process the message and determine what to do with it.
        """
        self.log.info("receiverId <%s> Received: <%s> " % (self.receiverId, msg['body']))
        
        #return super(MyStomp, self).ack(msg) 
        return stomper.NO_REPONSE_NEEDED