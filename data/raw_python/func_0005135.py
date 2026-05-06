def randbytes(self):
        """ -> #bytes result of bytes-encoded :func:gen_rand_str """
        return gen_rand_str(
            10, 30, use=self.random, keyspace=list(self.keyspace)
        ).encode("utf-8")