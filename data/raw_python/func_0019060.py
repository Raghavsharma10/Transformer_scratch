def pass_q_v1(self):
    """Update outflow."""
    sta = self.sequences.states.fastaccess
    out = self.sequences.outlets.fastaccess
    out.q[0] += sta.qa