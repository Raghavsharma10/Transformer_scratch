def create_api_context(self, cls):
        """Create and return an API context"""
        return self.api_context_schema().load({
            "name": cls.name,
            "cls": cls,
            "inst": [],
            "conf": self.conf.get_api_service(cls.name),
            "calls": self.conf.get_api_calls(),
            "shared": {},  # Used per-API to monitor state
            "log_level": self.conf.get_log_level(),
            "callback": self.receive
            })