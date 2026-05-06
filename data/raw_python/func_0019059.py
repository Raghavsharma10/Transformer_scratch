def pick_q_v1(self):
    """Update inflow."""
    sta = self.sequences.states.fastaccess
    inl = self.sequences.inlets.fastaccess
    sta.qz = 0.
    for idx in range(inl.len_q):
        sta.qz += inl.q[idx][0]