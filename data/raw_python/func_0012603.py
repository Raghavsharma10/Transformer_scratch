def add_field_like(self, name, like_array):
        """
        Add a new field to the Datamat with the dtype of the
        like_array and the shape of the like_array except for the first
        dimension which will be instead the field-length of this Datamat.
        """
        new_shape = list(like_array.shape)
        new_shape[0] = len(self)
        new_data = ma.empty(new_shape, like_array.dtype)
        new_data.mask = True
        self.add_field(name, new_data)