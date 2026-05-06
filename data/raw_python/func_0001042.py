def process(self, quoted=False):
        ''' Parse an URL '''
        self.p = urlparse(self.raw)
        self.scheme = self.p.scheme
        self.netloc = self.p.netloc
        self.opath = self.p.path if not quoted else quote(self.p.path)
        self.path = [x for x in self.opath.split('/') if x]
        self.params = self.p.params
        self.query = parse_qs(self.p.query, keep_blank_values=True)
        self.fragment = self.p.fragment