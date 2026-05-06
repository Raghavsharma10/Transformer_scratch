def stop(self, dummy_signum=None, dummy_frame=None):
        """ Shutdown process (this method is also a signal handler) """
        logging.info('Shutting down ...')
        self.socket.close()
        sys.exit(0)