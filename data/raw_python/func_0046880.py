def has_unlock_previous(self):
        """stub"""
        if 'unlockPrevious' not in self.my_osid_object._my_map or \
                self.my_osid_object._my_map['unlockPrevious'] is None:
            return False
        return True