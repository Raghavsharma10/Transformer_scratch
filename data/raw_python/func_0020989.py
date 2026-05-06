def connected(self, msg):
        """Once I've connected I want to subscribe to my the message queue.
        """
        stomper.Engine.connected(self, msg)

        self.log.info("Connected: session %s. Beginning say hello." % msg['headers']['session'])
        
        def setup_looping_call():
            lc = LoopingCall(self.send)
            lc.start(2)
            
        reactor.callLater(1, setup_looping_call)

        f = stomper.Frame()
        f.unpack(stomper.subscribe(DESTINATION))

        # ActiveMQ specific headers:
        #
        # prevent the messages we send comming back to us.
        f.headers['activemq.noLocal'] = 'true'
        
        return f.pack()