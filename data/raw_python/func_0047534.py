def set_face_values(self, front_face_value, side_face_value, top_face_value):
        """stub"""
        if front_face_value is None or side_face_value is None or top_face_value is None:
            raise NullArgument()
        self.add_integer_value(value=int(front_face_value), label='frontFaceValue')
        self.add_integer_value(value=int(side_face_value), label='sideFaceValue')
        self.add_integer_value(value=int(top_face_value), label='topFaceValue')