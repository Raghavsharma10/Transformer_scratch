def pick_input_v1(self):
    """Updates |Input| based on |Total|."""
    flu = self.sequences.fluxes.fastaccess
    inl = self.sequences.inlets.fastaccess
    flu.input = 0.
    for idx in range(inl.len_total):
        flu.input += inl.total[idx][0]