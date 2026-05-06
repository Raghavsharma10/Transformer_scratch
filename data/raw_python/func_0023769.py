def get_public_ip(self, addr_family=None, *args, **kwargs):
        """Alias for get_ip('public')"""
        return self.get_ip('public', addr_family, *args, **kwargs)