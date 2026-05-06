def set_euler_angle_values(self, x_angle, y_angle, z_angle):
        """stub"""
        if x_angle is None or y_angle is None or z_angle is None:
            raise NullArgument()
        self.add_integer_value(value=x_angle, label='xAngle')
        self.add_integer_value(value=y_angle, label='yAngle')
        self.add_integer_value(value=z_angle, label='zAngle')