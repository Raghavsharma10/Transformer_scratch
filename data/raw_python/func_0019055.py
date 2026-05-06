def calc_ag_v1(self):
    """Sum the through flown area of the total cross section.

    Required flux sequences:
      |AM|
      |AV|
      |AVR|

    Calculated flux sequence:
      |AG|

    Example:

        >>> from hydpy.models.lstream import *
        >>> parameterstep()
        >>> fluxes.am = 1.0
        >>> fluxes.av= 2.0, 3.0
        >>> fluxes.avr = 4.0, 5.0
        >>> model.calc_ag_v1()
        >>> fluxes.ag
        ag(15.0)
    """
    flu = self.sequences.fluxes.fastaccess
    flu.ag = flu.am+flu.av[0]+flu.av[1]+flu.avr[0]+flu.avr[1]