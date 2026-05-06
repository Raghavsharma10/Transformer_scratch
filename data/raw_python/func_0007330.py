def request(self, path, action, data=''):
        """To make a request to the API."""
        # Check if the path includes URL or not.
        head = self.base_url
        if path.startswith(head):
            path = path[len(head):]
            path = quote_plus(path, safe='/')
        if not path.startswith(self.api):
            path = self.api + path
        log.debug('Using path %s' % path)

        # If we have data, convert to JSON
        if data:
            data = json.dumps(data)
            log.debug('Data to sent: %s' % data)
        # In case of key authentication
        if self.private_key and self.public_key:
            timestamp = str(int(time.time()))
            log.debug('Using timestamp: {}'.format(timestamp))
            unhashed = path + timestamp + str(data)
            log.debug('Using message: {}'.format(unhashed))
            self.hash = hmac.new(str.encode(self.private_key),
                                 msg=unhashed.encode('utf-8'),
                                 digestmod=hashlib.sha256).hexdigest()
            log.debug('Authenticating with hash: %s' % self.hash)
            self.headers['X-Public-Key'] = self.public_key
            self.headers['X-Request-Hash'] = self.hash
            self.headers['X-Request-Timestamp'] = timestamp
            auth = False
        # In case of user credentials authentication
        elif self.username and self.password:
            auth = requests.auth.HTTPBasicAuth(self.username, self.password)
        # Set unlock reason
        if self.unlock_reason:
            self.headers['X-Unlock-Reason'] = self.unlock_reason
            log.info('Unlock Reason: %s' % self.unlock_reason)
        url = head + path
        # Try API request and handle Exceptions
        try:
            if action == 'get':
                log.debug('GET request %s' % url)
                self.req = requests.get(url, headers=self.headers, auth=auth,
                                        verify=False)
            elif action == 'post':
                log.debug('POST request %s' % url)
                self.req = requests.post(url, headers=self.headers, auth=auth,
                                         verify=False, data=data)
            elif action == 'put':
                log.debug('PUT request %s' % url)
                self.req = requests.put(url, headers=self.headers,
                                        auth=auth, verify=False,
                                        data=data)
            elif action == 'delete':
                log.debug('DELETE request %s' % url)
                self.req = requests.delete(url, headers=self.headers,
                                           verify=False, auth=auth)

            if self.req.content == b'':
                result = None
                log.debug('No result returned.')
            else:
                result = self.req.json()
                if 'error' in result and result['error']:
                    raise TPMException(result['message'])

        except requests.exceptions.RequestException as e:
            log.critical("Connection error for " + str(e))
            raise TPMException("Connection error for " + str(e))

        except ValueError as e:
            if self.req.status_code == 403:
                log.warning(url + " forbidden")
                raise TPMException(url + " forbidden")
            elif self.req.status_code == 404:
                log.warning(url + " forbidden")
                raise TPMException(url + " not found")
            else:
                message = ('%s: %s %s' % (e, self.req.url, self.req.text))
                log.debug(message)
                raise ValueError(message)

        return result