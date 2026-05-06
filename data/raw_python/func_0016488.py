def info(self):
        """Get connection info."""
        backend_cls = self.backend_cls or "amqplib"
        port = self.port or self.create_backend().default_port
        return {"hostname": self.hostname,
                "userid": self.userid,
                "password": self.password,
                "virtual_host": self.virtual_host,
                "port": port,
                "insist": self.insist,
                "ssl": self.ssl,
                "transport_cls": backend_cls,
                "backend_cls": backend_cls,
                "connect_timeout": self.connect_timeout}