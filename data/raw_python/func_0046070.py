def get_text(self, label):
        """stub"""
        if self.has_text(label):
            # Should this return a DisplayText?
            return DisplayText(self.my_osid_object._my_map['texts'][label])
        raise IllegalState()