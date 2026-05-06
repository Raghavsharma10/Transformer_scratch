def pic_inflow_v1(self):
    """Update the inlet link sequence.

    Required inlet sequence:
      |dam_inlets.Q|

    Calculated flux sequence:
      |Inflow|

    Basic equation:
      :math:`Inflow = Q`
    """
    flu = self.sequences.fluxes.fastaccess
    inl = self.sequences.inlets.fastaccess
    flu.inflow = inl.q[0]