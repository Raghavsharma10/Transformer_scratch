def ack(self, msg):
        """Override this and do some customer message handler.
        """
        print("Got a message:\n%s\n" % msg['body'])
        
        # do something with the message...
        
        # Generate the ack or not if you subscribed with ack='auto'
        return super(Pong, self).ack(msg)