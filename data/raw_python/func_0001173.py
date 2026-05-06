def get_config(self):
        """
        Sets up the basic config from the variables passed in
        all of these are from what Heroku gives you.
        """
        self.create_ssl_certs()

        config = {
            "bootstrap_servers": self.get_brokers(),
            "security_protocol": 'SSL',
            "ssl_cafile": self.ssl["ca"]["file"].name,
            "ssl_certfile": self.ssl["cert"]["file"].name,
            "ssl_keyfile": self.ssl["key"]["file"].name,
            "ssl_check_hostname": False,
            "ssl_password": None
        }
        self.config.update(config)