def clear_angle_values(self):
        """stub"""
        if (self.get_euler_rotation_values_metadata().is_read_only() or
                self.get_euler_rotation_values_metadata().is_required()):
            raise NoAccess()
        self.clear_integer_value('xAngle')
        self.clear_integer_value('yAngle')
        self.clear_integer_value('zAngle')