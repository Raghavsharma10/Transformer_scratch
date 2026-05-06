def check_channel_pty_request(self, channel, term, width, height, pixelwidth,
                                  pixelheight, modes):
        '''Request to allocate a PTY terminal.'''
        #self.sshterm = term
        #print "term: %r, modes: %r" % (term, modes)
        log.debug('PTY requested.  Setting up %r.', self.telnet_handler)
        pty_thread = Thread( target=self.start_pty_request, args=(channel, term, modes) )
        self.channels[channel] = pty_thread
        
        return True