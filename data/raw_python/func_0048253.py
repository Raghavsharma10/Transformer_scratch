def force_balanced_LP(self):
        r"""Force balanced lamination parameters

        The lamination parameters `\xi_{A2}` and `\xi_{A4}` are set to null
        to force a balanced laminate.

        """
        dummy, xiA1, xiA2, xiA3, xiA4 = self.xiA
        self.xiA = np.array([1, xiA1, 0, xiA3, 0], dtype=np.float64)
        self.calc_ABDE_from_lamination_parameters()