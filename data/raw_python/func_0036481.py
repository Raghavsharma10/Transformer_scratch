def current(cls):
        """
        Helper method for getting the current peer of whichever host we're
        running on.
        """
        name = socket.getfqdn()
        ip = socket.gethostbyname(name)

        return cls(name, ip)