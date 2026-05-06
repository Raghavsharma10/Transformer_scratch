def colRowIsOnSciencePixelList(self, col, row, padding=DEFAULT_PADDING):
        """similar to colRowIsOnSciencePixelList() but takes lists as input"""
        out = np.ones(len(col), dtype=bool)
        col_arr = np.array(col)
        row_arr = np.array(row)
        mask = np.bitwise_or(col_arr < 12. - padding, col_arr > 1111 + padding)
        out[mask] = False
        mask = np.bitwise_or(row_arr < 20. - padding, row_arr > 1043 + padding)
        out[mask] = False
        return out