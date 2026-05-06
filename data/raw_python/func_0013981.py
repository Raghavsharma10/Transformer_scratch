def policy(self):
        """Returns policy which contains this ioclass."""
        policies = VNXIOPolicy.get(cli=self._cli)
        ret = None
        for policy in policies:
            contained = policy.ioclasses.name
            if self._get_name() in contained:
                ret = VNXIOPolicy.get(name=policy.name, cli=self._cli)
                break
        return ret