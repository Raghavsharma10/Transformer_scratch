def check_internet_connection(self, remote_server: str = "api.isogeo.com") -> bool:
        """Test if an internet connection is operational.
        Src: http://stackoverflow.com/a/20913928/2556577.

        :param str remote_server: remote server used to check
        """
        try:
            # see if we can resolve the host name -- tells us if there is
            # a DNS listening
            host = socket.gethostbyname(remote_server)
            # connect to the host -- tells us if it's reachable
            sock = socket.create_connection((host, 80), 2)
            sock.close()
            return True
        except Exception as e:
            logging.error(e)
            return False