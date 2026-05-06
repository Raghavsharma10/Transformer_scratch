def has_max_attempts(self):
        """stub"""
        if 'maxAttempts' not in self.my_osid_object._my_map or \
                self.my_osid_object._my_map['maxAttempts'] is None:
            return False
        return True