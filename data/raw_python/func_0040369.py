def _worker_http(self, monitor):
        """
        Process an http monitor.
        """
        self.thread_debug("process_http", data=monitor, module='handler')
        query = monitor['query']
        method = query['method'].lower()
        self.stats.http_run += 1
        try:
            target = monitor['target']
            url = 'http://{host}:{port}{path}'.format(path=query['path'], **target)
            response = {
                'url': url,
                'status': 'failed',
                'result': {},
                'monitor': monitor,
                'message': 'did not meet expected result or no expected result defined',
                'elapsedms': monitor['timeout']*1000,
                'code':0
            }

            # not sed_env_dict -- we do not want to xref headers
            headers = query.get('headers', {})
            for elem in headers:
                headers[elem] = self.sed_env(headers[elem], {}, '')

            res = response['result'] = getattr(requests, method)(url,
                                                                 headers=headers,
                                                                 timeout=monitor['timeout'])
            response['code'] = res.status_code
            response['elapsedms'] = res.elapsed.total_seconds() * 1000
            if 'response-code' in monitor['expect']:
                if int(monitor['expect']['response-code']) == res.status_code:
                    response['message'] = ''
                    response['status'] = 'ok'
                else: # abort with failure, do not pass go
                    return response

            if 'content' in monitor['expect']:
                if monitor['expect']['content'] in res.text:
                    response['message'] = ''
                    response['status'] = 'ok'
                else: # abort with failure, do not pass go
                    return response

            if 'regex' in monitor['expect']:
                if re.search(monitor['expect']['regex'], res.text):
                    response['message'] = ''
                    response['status'] = 'ok'
                else: # abort with failure, do not pass go
                    return response

        except requests.exceptions.Timeout:
            response['message'] = 'timeout'
        except requests.exceptions.ConnectionError:
            response['message'] = 'connect-failed'
            response['elapsedms'] = -1
        return response