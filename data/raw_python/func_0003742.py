def active_inactive(self):
        """The indexes of the active and the inactive cell vectors"""
        active_indices = []
        inactive_indices = []
        for index, active in enumerate(self.active):
            if active:
                active_indices.append(index)
            else:
                inactive_indices.append(index)
        return active_indices, inactive_indices