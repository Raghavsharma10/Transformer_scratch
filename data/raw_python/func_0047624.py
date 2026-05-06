def get_solution(self, parameters=None):
        """stub"""
        if not self.has_solution():
            raise IllegalState()
        return DisplayText(self.my_osid_object._my_map['solution'])