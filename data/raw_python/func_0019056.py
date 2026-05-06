def calc_qg_v1(self):
    """Calculate the discharge of the total cross section.

    Method |calc_qg_v1| applies the actual versions of all methods for
    calculating the flown through areas, wetted perimeters and discharges
    of the different cross section compartments.  Hence its requirements
    might be different for various application models.
    """
    flu = self.sequences.fluxes.fastaccess
    self.calc_am_um()
    self.calc_qm()
    self.calc_av_uv()
    self.calc_qv()
    self.calc_avr_uvr()
    self.calc_qvr()
    flu.qg = flu.qm+flu.qv[0]+flu.qv[1]+flu.qvr[0]+flu.qvr[1]