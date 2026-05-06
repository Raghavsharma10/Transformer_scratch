def randtld(self):
        """ -> a random #str tld via :mod:tlds """
        self.tlds = tuple(tlds.tlds) if not self.tlds else self.tlds
        return self.random.choice(self.tlds)