def pass_requiredremotesupply_v1(self):
    """Update the outlet link sequence |dam_outlets.S|."""
    flu = self.sequences.fluxes.fastaccess
    sen = self.sequences.senders.fastaccess
    sen.s[0] += flu.requiredremotesupply