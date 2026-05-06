def randurl(self):
        """ -> a random url-like #str via :prop:randdomain, :prop:randtld,
                and :prop:randpath
        """
        return "{}://{}.{}/{}".format(
            self.random.choice(("http", "https")),
            self.randdomain, self.randtld, self.randpath)