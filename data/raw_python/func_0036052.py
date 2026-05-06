def todict(self):
        """Returns a dictionary fully representing the state of this object
        """
        return {'index': self.index,
                'seed': hb_encode(self.seed),
                'n': self.n,
                'root': hb_encode(self.root),
                'hmac': hb_encode(self.hmac),
                'timestamp': self.timestamp}