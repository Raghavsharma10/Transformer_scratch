def check_time(self):
        """ Make sure our Honeypot time is consistent, and not too far off
        from the actual time. """

        poll = self.config['timecheck']['poll']
        ntp_poll = self.config['timecheck']['ntp_pool']
        while True:
            clnt = ntplib.NTPClient()
            try:
                response = clnt.request(ntp_poll, version=3)
                diff = response.offset
                if abs(diff) >= 15:
                    logger.error('Timings found to be far off, shutting down drone ({0})'.format(diff))
                    sys.exit(1)
                else:
                    logger.debug('Polled ntp server and found that drone has {0} seconds offset.'.format(diff))
            except (ntplib.NTPException, _socket.error) as ex:
                logger.warning('Error while polling ntp server: {0}'.format(ex))
            gevent.sleep(poll * 60 * 60)