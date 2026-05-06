def _obtain_new_ip(self):
        """
        Change Tor's IP.
        """
        with Controller.from_port(
            address=self.tor_address, port=self.tor_port
        ) as controller:
            controller.authenticate(password=self.tor_password)
            controller.signal(Signal.NEWNYM)

        # Wait till the IP 'settles in'.
        sleep(0.5)