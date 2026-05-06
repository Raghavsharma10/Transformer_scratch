def login(self):
        """
        Try to login and set the internal session id.

        Please note:
        - Any failed login resets all existing session ids, even of
          other users.
        - SIDs expire after some time
        """
        response = self.session.get(self.base_url + '/login_sid.lua', timeout=10)
        xml = ET.fromstring(response.text)
        if xml.find('SID').text == "0000000000000000":
            challenge = xml.find('Challenge').text
            url = self.base_url + "/login_sid.lua"
            response = self.session.get(url, params={
                "username": self.username,
                "response": self.calculate_response(challenge, self.password),
            }, timeout=10)
            xml = ET.fromstring(response.text)
            sid = xml.find('SID').text
            if xml.find('SID').text == "0000000000000000":
                blocktime = int(xml.find('BlockTime').text)
                exc = Exception("Login failed, please wait {} seconds".format(
                    blocktime
                ))
                exc.blocktime = blocktime
                raise exc
            self.sid = sid
            return sid