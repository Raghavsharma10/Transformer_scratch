def GaP_Eg_Gamma(self, **kwargs):
    '''
    Returns the Gamma-valley bandgap, Eg_Gamma, in electron Volts at a
    given temperature, T, in Kelvin (default: 300 K).

    GaP has a unique Gamma-gap temperature dependence.
    '''
    T = kwargs.get('T', 300.)
    if T < 1e-4:
        return self.Eg_Gamma_0()
    return self.Eg_Gamma_0() + 0.1081 * (1 - 1. / tanh(164. / T))  # eV