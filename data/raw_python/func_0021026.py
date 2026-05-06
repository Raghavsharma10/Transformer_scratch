def _findMessageBytes ( self, data ):
        """
        I examine the data passed to me and return a 2-tuple of the form:
        
          ( message_length, header_length )
          
        where message_length is the length in bytes of the first complete
        message, if it contains at least one message, or 0 if it
        contains no message.
        
        If message_length is non-zero, header_length contains the length in
        bytes of the header. If message_length is zero, header_length should
        be ignored.

        You should probably not call me directly. Call getOneMessage instead.
        """

        # Sanity check. See the docstring for the method to see what it
        # does an why we need it.
        self.syncBuffer()
        
        # If the string '\n\n' does not exist, we don't even have the complete
        # header yet and we MUST exit.
        try:
            i = data.index ( '\n\n' )
        except ValueError:
            return ( 0, 0 )
        # If the string '\n\n' exists, then we have the entire header and can
        # check for the content-length header. If it exists, we can check
        # the length of the buffer for the number of bytes, else we check for
        # the existence of a null byte.

        # Pull out the header before we perform the regexp search. This
        # prevents us from matching (possibly malicious) strings in the
        # body.
        _hdr = self.buffer[:i]
        match = content_length_re.search ( _hdr )
        if match:
            # There was a content-length header, so read out the value.
            content_length = int ( match.groups()[0] )

            # THIS IS NO LONGER THE CASE IF WE REMOVE THE '\n\n' in
            # Frame.pack()
            
            # This is the content length of the body up until the null
            # byte, not the entire message. Note that this INCLUDES the 2
            # '\n\n' bytes inserted by the STOMP encoder after the body
            # (see the calculation of content_length in
            # StompEngine.callRemote()), so we only need to add 2 final bytes
            # for the footer.
            #
            #The message looks like:
            #
            #   <header>\n\n<body>\n\n\x00\n
            #           ^         ^^^^
            #          (i)         included in content_length!
            #
            # We have the location of the end of the header (i), so we
            # need to ensure that the message contains at least:
            #
            #     i + len ( '\n\n' ) + content_length + len ( '\x00\n' )
            #
            # Note that i is also the count of bytes in the header, because
            # of the fact that str.index() returns a 0-indexed value.
            req_len = i + len_sep + content_length + len_footer
            # log.msg ( "We have [%s] bytes and need [%s] bytes" %
            #           ( len ( data ), req_len, ) )
            if len ( data ) < req_len:
                # We don't have enough bytes in the buffer.
                return ( 0, 0 )
            else:
                # We have enough bytes in the buffer
                return ( req_len, i )
        else:
            # There was no content-length header, so just look for the
            # message terminator ('\x00\n' ).
            try:
                j = data.index ( '\x00\n' )
            except ValueError:
                return ( 0, 0 )
            # j points to the 0-indexed location of the null byte. However,
            # we need to add 1 (to turn it into a byte count) and 1 to take
            # account of the final '\n' character after the null byte.
            return ( j + 2, i )