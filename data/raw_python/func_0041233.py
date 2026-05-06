def _run__http(self, action, replace):
        """More complex HTTP query."""

        query = action['query']
#        self._debug = True
        url = '{type}://{host}{path}'.format(path=query['path'], **action)
        content = None
        method = query.get('method', "get").lower()
        self.debug("{} {} url={}\n", action['type'], method, url)
        if method == "post":
            content = query['content']
        headers = query.get('headers', {})

        if replace and action.get('template'):
            self.rfxcfg.macro_expand(url, replace)
            if content:
                if isinstance(content, dict):
                    for key, value in content.items():
                        content[key] = self.rfxcfg.macro_expand(value, replace)
                else:
                    content = self.rfxcfg.macro_expand(content, replace)

            newhdrs = dict()
            for key, value in headers.items():
                newhdrs[key.lower()] = self.rfxcfg.macro_expand(value, replace)
            headers = newhdrs

        self.debug("{} headers={}\n", action['type'], headers)
        self.debug("{} content={}\n", action['type'], content)

        if content and isinstance(content, dict):
            content = json.dumps(content)

        self.logf("Action {name} {type}\n", **action)
        result = getattr(requests, method)(url, headers=headers, data=content, timeout=action.get('timeout', 5))
        expect = action.get('expect', {})
        expected_codes = expect.get("response-codes", (200, 201, 202, 204))
        self.debug("{} expect codes={}\n", action['type'], expected_codes)
        self.debug("{} status={} content={}\n", action['type'], result.status_code, result.text)
        if result.status_code not in expected_codes:
            self.die("Unable to make {} call, unexpected result ({})",
                     action['type'], result.status_code)

        if 'content' in expect:
            self.debug("{} expect content={}\n", action['type'], expect['content'])
            if expect['content'] not in result.text:
                self.die("{} call to {} failed\nExpected: {}\nReceived:\n{}",
                         action['type'], url, expect['content'], result.text)

        if 'regex' in expect:
            self.debug("{} expect regex={}\n", action['type'], expect['regex'])
            if not re.search(expect['regex'], result.text):
                self.die("{} call to {} failed\nRegex: {}\nDid not match:\n{}",
                         action['type'], url, expect['regex'], result.text)

        self.log(result.text, level=common.log_msg)

        self.logf("Success, status={}\n", result.status_code, level=common.log_good)
        return True