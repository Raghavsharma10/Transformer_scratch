def init(self):
        """Prepare the HTTP handler, URL, and HTTP headers for all subsequent requests"""
        self.debug('Initializing %r', self)
        proto = self.server.split('://')[0]

        if proto == 'https':
            if hasattr(ssl, 'create_default_context'):
                context = ssl.create_default_context()

                if self.ssl_verify:
                    context.check_hostname = True
                    context.verify_mode = ssl.CERT_REQUIRED
                else:
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_NONE

                self._http_handler = urllib2.HTTPSHandler(debuglevel=0, context=context)
            else:
                self._http_handler = urllib2.HTTPSHandler(debuglevel=0)
        elif proto == 'http':
            self._http_handler = urllib2.HTTPHandler(debuglevel=0)
        else:
            raise ValueError('Invalid protocol %s' % proto)

        self._api_url = self.server + '/api_jsonrpc.php'
        self._http_headers = {
            'Content-Type': 'application/json-rpc',
            'User-Agent': 'python/zabbix_api',
        }

        if self.httpuser:
            self.debug('HTTP authentication enabled')
            auth = self.httpuser + ':' + self.httppasswd
            self._http_headers['Authorization'] = 'Basic ' + b64encode(auth.encode('utf-8')).decode('ascii')