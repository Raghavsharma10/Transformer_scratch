def pass_outputs_v1(self):
    """Updates |Branched| based on |Outputs|."""
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    out = self.sequences.outlets.fastaccess
    for bdx in range(der.nmbbranches):
        out.branched[bdx][0] += flu.outputs[bdx]