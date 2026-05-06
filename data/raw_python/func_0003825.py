def from_coordinates(cls, coordinates, labels):
        """Initialize a similarity descriptor

           Arguments:
             coordinates  --  a Nx3 numpy array
             labels  --  a list with integer labels used to identify atoms of
                         the same type
        """
        from molmod.ext import molecules_distance_matrix
        distance_matrix = molecules_distance_matrix(coordinates)
        return cls(distance_matrix, labels)