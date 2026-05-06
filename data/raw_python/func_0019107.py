def pass_q_v1(self):
    """Update the outlet link sequence."""
    flu = self.sequences.fluxes.fastaccess
    out = self.sequences.outlets.fastaccess
    out.q[0] += flu.qa