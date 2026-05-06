def pick_q_v1(self):
    """Update inflow."""
    flu = self.sequences.fluxes.fastaccess
    inl = self.sequences.inlets.fastaccess
    flu.qin = 0.
    for idx in range(inl.len_q):
        flu.qin += inl.q[idx][0]