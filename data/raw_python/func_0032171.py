def parse_response(self, response, header=None):
        """Parses the response message.

        The following graph shows the structure of response messages.

        ::

                                                        +----------+
                                                     +--+ data sep +<-+
                                                     |  +----------+  |
                                                     |                |
                  +--------+        +------------+   |    +------+    |
              +-->| header +------->+ header sep +---+--->+ data +----+----+
              |   +--------+        +------------+        +------+         |
              |                                                            |
            --+                                         +----------+       +-->
              |                                      +--+ data sep +<-+    |
              |                                      |  +----------+  |    |
              |                                      |                |    |
              |                                      |    +------+    |    |
              +--------------------------------------+--->+ data +----+----+
                                                          +------+

       """
        response = response.decode(self.encoding)
        if header:
            header = "".join((self.resp_prefix, header, self.resp_header_sep))
            if not response.startswith(header):
                raise IEC60488.ParsingError('Response header mismatch')
            response = response[len(header):]
        return response.split(self.resp_data_sep)