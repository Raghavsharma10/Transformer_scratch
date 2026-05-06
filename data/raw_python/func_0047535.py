def clear_face_values(self):
        """stub"""
        if (self.get_face_values_metadata().is_read_only() or
                self.get_face_values_metadata().is_required()):
            raise NoAccess()
        self.clear_integer_value('frontFaceValue')
        self.clear_integer_value('sideFaceValue')
        self.clear_integer_value('topFaceValue')