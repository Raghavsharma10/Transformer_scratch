def ack(self, msg):
        """Processes the received message. I don't need to 
        generate an ack message.
        
        """
        self.log.info("senderID:%s Received: %s " % (self.senderID, msg['body']))
        return stomper.NO_REPONSE_NEEDED