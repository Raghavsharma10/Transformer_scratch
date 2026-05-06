def pass_q_v1(self):
    """Assing the actual value of the lower joint of of the subreach
    downstream to the outlet sequence."""
    der = self.parameters.derived.fastaccess
    new = self.sequences.states.fastaccess_new
    out = self.sequences.outlets.fastaccess
    out.q[0] += new.qjoints[der.nmbsegments]