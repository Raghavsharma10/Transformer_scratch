def check_channel_shell_request(self, channel):
        '''Request to start a shell on the given channel'''
        try:
            self.channels[channel].start()
        except KeyError:
            log.error('Requested to start a channel (%r) that was not previously set up.', channel)
            return False
        else:
            return True