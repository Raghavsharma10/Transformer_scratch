def render_GET(self, request):
        """
        Begin sending the contents of this L{File} (or a subset of the
        contents, based on the 'range' header) to the given request.
        """
        request.setHeader(b'accept-ranges', b'bytes')

        producer = self.makeProducer(request, self.fileObject)

        if request.method == b'HEAD':
            return b''

        def done(ign):
            producer.stopProducing()

        request.notifyFinish().addCallbacks(done, done)
        producer.start()
        # and make sure the connection doesn't get closed
        return server.NOT_DONE_YET