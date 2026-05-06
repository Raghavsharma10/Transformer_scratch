def randpath(self):
        """ -> a random URI-like #str path """
        return '/'.join(
            gen_rand_str(3, 10, use=self.random, keyspace=list(self.keyspace))
            for _ in range(self.random.randint(0, 3)))