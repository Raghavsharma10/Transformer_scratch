def todict(self):
        """Returns a dictionary fully representing the state of this object
        """
        return {"f_key": hb_encode(self.f_key),
                "alpha_key": hb_encode(self.alpha_key),
                "chunks": self.chunks,
                "encrypted": self.encrypted,
                "iv": hb_encode(self.iv),
                "hmac": hb_encode(self.hmac)}