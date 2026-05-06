def pass_q_v1(self):
    """Update the outlet link sequence.

    Required derived parameter:
      |QFactor|

    Required flux sequences:
      |lland_fluxes.Q|

    Calculated flux sequence:
      |lland_outlets.Q|

    Basic equation:
       :math:`Q_{outlets} = QFactor \\cdot Q_{fluxes}`
    """
    der = self.parameters.derived.fastaccess
    flu = self.sequences.fluxes.fastaccess
    out = self.sequences.outlets.fastaccess
    out.q[0] += der.qfactor*flu.q