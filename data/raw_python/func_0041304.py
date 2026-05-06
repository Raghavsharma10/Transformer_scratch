def recv(self):
        """Returns the reply message or None if there was no reply."""
        try:
            items = self.poller.poll(self.timeout)
        except KeyboardInterrupt:
            return  # interrupted

        if items:
            # if we got a reply, process it
            msg = self.client.recv_multipart()
            self.close()
            if self.verbose:
                logging.info("I: received reply:")
                dump(msg)

            # Don't try to handle errors, just assert noisily
            assert len(msg) >= 4

            #first drop will be drop (cause empty)
            header = msg.pop(0)
            header = msg.pop(0)
            assert MDP.C_CLIENT == header

            #this one contains servicename
            #TODO: exploit this
            header = msg.pop(0)

            return msg
        else:
            logging.warn("W: permanent error, abandoning request")