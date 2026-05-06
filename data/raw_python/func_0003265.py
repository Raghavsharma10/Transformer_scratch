def quit(self, daemononly = False):
        '''
        Send quit event to quit the main loop
        '''
        if not self.quitting:
            self.quitting = True
            self.queue.append(SystemControlEvent(SystemControlEvent.QUIT, daemononly = daemononly), True)