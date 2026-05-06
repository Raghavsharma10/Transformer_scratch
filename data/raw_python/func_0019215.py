def pass_allowedremoterelieve_v1(self):
    """Update the outlet link sequence |dam_outlets.R|."""
    flu = self.sequences.fluxes.fastaccess
    sen = self.sequences.senders.fastaccess
    sen.r[0] += flu.allowedremoterelieve