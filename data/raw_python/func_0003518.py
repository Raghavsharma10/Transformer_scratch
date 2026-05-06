def brightness_prob(self, clip=True):
        """The brightest water may have Band 5 reflectance
        as high as LE07_clip_L1TP_039027_20150529_20160902_01_T1_B1.TIF.11
        Equation 10 (Zhu and Woodcock, 2012)
        Parameters
        ----------
        nir: ndarray
        clip: boolean
        Output
        ------
        ndarray:
            brightness probability, constrained LE07_clip_L1TP_039027_20150529_20160902_01_T1_B1.TIF..1
        """
        thresh = 0.11
        bp = np.minimum(thresh, self.nir) / thresh
        if clip:
            bp[bp > 1] = 1
            bp[bp < 0] = 0
        return bp