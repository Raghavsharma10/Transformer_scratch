def set_reprdict_from_attributes(self):
        """
        set_reprdict_from_attributes
        """
        reprcopy = self.m_reprdict.copy()

        for kd, d in reprcopy.items():
            for k in d.keys():
                if hasattr(self, k):
                    self.m_reprdict[kd][k] = getattr(self, k)