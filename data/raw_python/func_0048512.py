def set_custom(self,gmin,gmu,gsigma):
    """Set a minimum lengtha, and then the gaussian distribution parameters for cutting
       For any sequence longer than the minimum the guassian parameters will be used"""
    self._gauss_min = gmin
    self._gauss_mu = gmu
    self._gauss_sigma = gsigma