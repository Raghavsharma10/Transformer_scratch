def randstr(self):
        """ -> #str result of :func:gen_rand_str """
        return gen_rand_str(
            4, 10, use=self.random, keyspace=list(string.ascii_letters))