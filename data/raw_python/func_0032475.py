def parse(msg):
        """
        Helper method for parsing a Mongrel2 request string and returning a new
        `MongrelRequest` instance.
        """
        sender, conn_id, path, rest = msg.split(' ', 3)
        headers, rest = tnetstring.pop(rest)
        body, _ = tnetstring.pop(rest)

        if type(headers) is str:
            headers = json.loads(headers)

        return MongrelRequest(sender, conn_id, path, headers, body)