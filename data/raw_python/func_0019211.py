def pass_outflow_v1(self):
    """Update the outlet link sequence |dam_outlets.Q|."""
    flu = self.sequences.fluxes.fastaccess
    out = self.sequences.outlets.fastaccess
    out.q[0] += flu.outflow