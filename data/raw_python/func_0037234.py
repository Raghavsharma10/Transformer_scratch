def matches(self, a, b, **config):
        """ The message must match by package """
        package_a = self.processor._u2p(a['msg']['update']['title'])[0]
        package_b = self.processor._u2p(b['msg']['update']['title'])[0]
        if package_a != package_b:
            return False
        return True