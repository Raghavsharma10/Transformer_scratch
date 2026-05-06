def pass_actualremoterelease_v1(self):
    """Update the outlet link sequence |dam_outlets.S|."""
    flu = self.sequences.fluxes.fastaccess
    out = self.sequences.outlets.fastaccess
    out.s[0] += flu.actualremoterelease