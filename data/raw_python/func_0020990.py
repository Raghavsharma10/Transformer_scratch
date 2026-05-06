def send(self):
        """Send out a hello message periodically.
        """
        self.log.info("Saying hello (%d)." % self.counter)

        f = stomper.Frame()
        f.unpack(stomper.send(DESTINATION, 'hello there (%d)' % self.counter))

        self.counter += 1        

        # ActiveMQ specific headers:
        #
        #f.headers['persistent'] = 'true'

        self.transport.write(f.pack())