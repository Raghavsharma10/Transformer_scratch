def cross(self, vector):
        """Return a Vector instance as the cross product of two vectors"""
        return Vector((self.y * vector.z - self.z * vector.y),
                      (self.z * vector.x - self.x * vector.z),
                      (self.x * vector.y - self.y * vector.x))