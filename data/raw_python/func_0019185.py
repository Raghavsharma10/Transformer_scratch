def pic_inflow_v2(self):
    """Update the inlet link sequences.

    Required inlet sequences:
      |dam_inlets.Q|
      |dam_inlets.S|
      |dam_inlets.R|

    Calculated flux sequence:
      |Inflow|

    Basic equation:
      :math:`Inflow = Q + S + R`
    """
    flu = self.sequences.fluxes.fastaccess
    inl = self.sequences.inlets.fastaccess
    flu.inflow = inl.q[0]+inl.s[0]+inl.r[0]