def syncBuffer( self ):
        """
        I detect and correct corruption in the buffer.
        
        Corruption in the buffer is defined as the following conditions
        both being true:
        
            1. The buffer contains at least one newline;
            2. The text until the first newline is not a STOMP command.
        
        In this case, we heuristically try to flush bits of the buffer until
        one of the following conditions becomes true:
        
            1. the buffer starts with a STOMP command;
            2. the buffer does not contain a newline.
            3. the buffer is empty;

        If the buffer is deemed corrupt, the first step is to flush the buffer
        up to and including the first occurrence of the string '\x00\n', which
        is likely to be a frame boundary.

        Note that this is not guaranteed to be a frame boundary, as a binary
        payload could contain the string '\x00\n'. That condition would get
        handled on the next loop iteration.
        
        If the string '\x00\n' does not occur, the entire buffer is cleared.
        An earlier version progressively removed strings until the next newline,
        but this gets complicated because the body could contain strings that
        look like STOMP commands.
        
        Note that we do not check "partial" strings to see if they *could*
        match a command; that would be too resource-intensive. In other words,
        a buffer containing the string 'BUNK' with no newline is clearly
        corrupt, but we sit and wait until the buffer contains a newline before
        attempting to see if it's a STOMP command.
        """
        while True:
            if not self.buffer:
                # Buffer is empty; no need to do anything.
                break
            m = command_re.match ( self.buffer )
            if m is None:
                # Buffer doesn't even contain a single newline, so we can't
                # determine whether it's corrupt or not. Assume it's OK.
                break
            cmd = m.groups()[0]
            if cmd in stomper.VALID_COMMANDS:
                # Good: the buffer starts with a command.
                break
            else:
                # Bad: the buffer starts with bunk, so strip it out. We first
                # try to strip to the first occurrence of '\x00\n', which
                # is likely to be a frame boundary, but if this fails, we
                # strip until the first newline.
                ( self.buffer, nsubs ) = sync_re.subn ( '', self.buffer )

                if nsubs:
                    # Good: we managed to strip something out, so restart the
                    # loop to see if things look better.
                    continue
                else:
                    # Bad: we failed to strip anything out, so kill the
                    # entire buffer. Since this resets the buffer to a
                    # known good state, we can break out of the loop.
                    self.buffer = ''
                    break