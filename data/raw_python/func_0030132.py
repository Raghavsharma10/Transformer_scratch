def sub_description(self):
        """Time and space dscription"""
        gd = self.geo_description
        td = self.time_description

        if gd and td:
            return '{}, {}. {} Rows.'.format(gd, td, self._p.count)
        elif gd:
            return '{}. {} Rows.'.format(gd, self._p.count)
        elif td:
            return '{}. {} Rows.'.format(td, self._p.count)
        else:
            return '{} Rows.'.format(self._p.count)