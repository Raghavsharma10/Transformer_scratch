def get_private_ip(self, addr_family=None, *args, **kwargs):
        """Alias for get_ip('private')"""
        return self.get_ip('private', addr_family, *args, **kwargs)