def nhapDaiHan(self, cucSo, gioiTinh):
        """Nhap dai han

        Args:
            cucSo (TYPE): Description
            gioiTinh (TYPE): Description

        Returns:
            TYPE: Description
        """
        for cung in self.thapNhiCung:
            khoangCach = khoangCachCung(cung.cungSo, self.cungMenh, gioiTinh)
            cung.daiHan(cucSo + khoangCach * 10)
        return self