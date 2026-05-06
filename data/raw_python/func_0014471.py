def _compute_key(self, id, nbytes):
        "id is 'A' - 'F' for the various keys used by ssh"
        m = Message()
        m.add_mpint(self.K)
        m.add_bytes(self.H)
        m.add_byte(id)
        m.add_bytes(self.session_id)
        out = sofar = SHA.new(str(m)).digest()
        while len(out) < nbytes:
            m = Message()
            m.add_mpint(self.K)
            m.add_bytes(self.H)
            m.add_bytes(sofar)
            digest = SHA.new(str(m)).digest()
            out += digest
            sofar += digest
        return out[:nbytes]