def register_service(self, **kwargs):
        """
        register this service with consul
        kwargs passed to Consul.agent.service.register
        """
        kwargs.setdefault('name', self.app.name)
        self.session.agent.service.register(**kwargs)