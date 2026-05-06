def pass_actualremoterelieve_v1(self):
    """Update the outlet link sequence |dam_outlets.R|."""
    flu = self.sequences.fluxes.fastaccess
    out = self.sequences.outlets.fastaccess
    out.r[0] += flu.actualremoterelieve