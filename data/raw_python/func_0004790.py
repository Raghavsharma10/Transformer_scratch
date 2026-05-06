def calc_mass_2(mh,cm,nm,teff,logg):
    """ Table A2 in Martig 2016 """
    CplusN = calc_sum(mh,cm,nm)
    t = teff/4000.
    return (95.8689 - 10.4042*mh - 0.7266*mh**2
            + 41.3642*cm - 5.3242*cm*mh - 46.7792*cm**2
            + 15.0508*nm - 0.9342*nm*mh - 30.5159*nm*cm - 1.6083*nm**2
            - 67.6093*CplusN + 7.0486*CplusN*mh + 133.5775*CplusN*cm + 38.9439*CplusN*nm - 88.9948*CplusN**2
            - 144.1765*t + 5.1180*t*mh - 73.7690*t*cm - 15.2927*t*nm + 101.7482*t*CplusN + 27.7690*t**2
            - 9.4246*logg + 1.5159*logg*mh + 16.0412*logg*cm + 1.3549*logg*nm - 18.6527*logg*CplusN + 28.8015*logg*t - 4.0982*logg**2)