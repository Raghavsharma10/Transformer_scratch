def _append(self, signature, fields=(), response=None):
        """ Add a message to the outgoing queue.

        :arg signature: the signature of the message
        :arg fields: the fields of the message as a tuple
        :arg response: a response object to handle callbacks
        """
        self.packer.pack_struct(signature, fields)
        self.output_buffer.chunk()
        self.output_buffer.chunk()
        self.responses.append(response)