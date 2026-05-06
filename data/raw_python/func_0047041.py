def _curl_bitmex(self, api, query=None, postdict=None, timeout=3, verb=None):
        """Send a request to BitMEX Servers."""
        # Handle URL
        url = self.base_url + api

        # Default to POST if data is attached, GET otherwise
        if not verb:
            verb = 'POST' if postdict else 'GET'

        # Auth: Use Access Token by default, API Key/Secret if provided
        auth = AccessTokenAuth(self.token)
        if self.apiKey:
            auth = APIKeyAuthWithExpires(self.apiKey, self.apiSecret)

        # Make the request
        try:
#            url = "http://httpbin.org/post"
            req = requests.Request(verb, url, data=postdict, auth=auth, params=query)
            prepped = self.session.prepare_request(req)
            response = self.session.send(prepped, timeout=timeout)
            # Make non-200s throw
            response.raise_for_status()

        except requests.exceptions.HTTPError as e:
            # 401 - Auth error. Re-auth and re-run this request.
            if response.status_code == 401:
                if self.token is None:
                    self.logger.error("Login information or API Key incorrect, please check and restart.")
                    self.logger.error("Error: " + response.text)
                    if postdict:
                        self.logger.error(postdict)
                self.logger.warning("Token expired, reauthenticating...")
                sleep(1)
                self.authenticate()
                return self._curl_bitmex(api, query, postdict, timeout, verb)

            # 404, can be thrown if order canceled does not exist.
            elif response.status_code == 404:
                if verb == 'DELETE':
                    self.logger.error("Order not found: %s" % postdict['orderID'])
                    return
                self.logger.error("Unable to contact the BitMEX API (404). " +
                                  "Request: %s \n %s" % (url, json.dumps(postdict)))
            # 429, ratelimit
            elif response.status_code == 429:
                self.logger.error("Ratelimited on current request. Sleeping, then trying again. Try fewer " +
                                  "order pairs or contact support@bitmex.com to raise your limits. " +
                                  "Request: %s \n %s" % (url, json.dumps(postdict)))
                sleep(1)
                return self._curl_bitmex(api, query, postdict, timeout, verb)

            # 503 - BitMEX temporary downtime, likely due to a deploy. Try again
            elif response.status_code == 503:
                self.logger.warning("Unable to contact the BitMEX API (503), retrying. " +
                                    "Request: %s \n %s" % (url, json.dumps(postdict)))
                sleep(1)
                return self._curl_bitmex(api, query, postdict, timeout, verb)
            # Unknown Error
            else:
                self.logger.error("Unhandled Error: %s: %s %s" % (e, response.text, json.dumps(response.json(), indent=4)))
                self.logger.error("Endpoint was: %s %s" % (verb, api))
        except requests.exceptions.Timeout as e:
            # Timeout, re-run this request
            self.logger.warning("Timed out, retrying...")
            return self._curl_bitmex(api, query, postdict, timeout, verb)

        except requests.exceptions.ConnectionError as e:
            self.logger.warning("Unable to contact the BitMEX API (ConnectionError). Please check the URL. Retrying. " +
                                "Request: %s \n %s" % (url, json.dumps(postdict)))
            sleep(1)
            return self._curl_bitmex(api, query, postdict, timeout, verb)

        return response.json()