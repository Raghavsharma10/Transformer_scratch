def _handler_http(self, result):
        """
        Handle the result of an http monitor
        """
        monitor = result['monitor']
        self.thread_debug("process_http", data=monitor, module='handler')
        self.stats.http_handled += 1

        # splunk will pick this up
        logargs = {
            'type':"metric",
            'endpoint': result['url'],
            'pipeline': monitor['pipeline'],
            'service': monitor['service'],
            'instance': monitor['instance'],
            'status': result['status'],
            'elapsed-ms': round(result['elapsedms'], 5),
            'code': result['code']
        }
        self.NOTIFY(result['message'], **logargs)

        # if our status has changed, also update Reflex Engine
        if result['status'] != self.instances[monitor['instance']]['status']:
            # do some retry/counter steps on failure?
            self.instances[monitor['instance']]['status'] = result['status']
            self.rcs.patch('instance',
                           monitor['instance'],
                           {'status': result['status']})