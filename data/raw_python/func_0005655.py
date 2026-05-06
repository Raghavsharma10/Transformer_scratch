def charge(self, data):
        """Second stage of an OPR request"""
        token = data.get("token", self._response["token"])
        data = {
            "token": token,
            "confirm_token": data.get("confirm_token")
        }
        return self._process('opr/charge', data)