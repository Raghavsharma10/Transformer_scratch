def bind(self, data):
        """ Bind a VertexBuffer that has structured data
        
        Parameters
        ----------
        data : VertexBuffer
            The vertex buffer to bind. The field names of the array
            are mapped to attribute names in GLSL.
        """
        # Check
        if not isinstance(data, VertexBuffer):
            raise ValueError('Program.bind() requires a VertexBuffer.')
        # Apply
        for name in data.dtype.names:
            self[name] = data[name]