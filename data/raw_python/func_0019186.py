def pic_totalremotedischarge_v1(self):
    """Update the receiver link sequence."""
    flu = self.sequences.fluxes.fastaccess
    rec = self.sequences.receivers.fastaccess
    flu.totalremotedischarge = rec.q[0]