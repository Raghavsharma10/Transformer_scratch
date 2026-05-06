def connected(self, msg):
        """Once I've connected I want to subscribe to my the message queue.
        """
        super(MyStomp, self).connected(msg)

        self.log.info("connected: session %s" % msg['headers']['session'])
        f = stomper.Frame()
        f.unpack(stomper.subscribe(DESTINATION))
        return f.pack()