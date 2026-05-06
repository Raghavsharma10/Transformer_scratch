def pick_q_v1(self):
    """Assign the actual value of the inlet sequence to the upper joint
    of the subreach upstream."""
    inl = self.sequences.inlets.fastaccess
    new = self.sequences.states.fastaccess_new
    new.qjoints[0] = 0.
    for idx in range(inl.len_q):
        new.qjoints[0] += inl.q[idx][0]