def inverse(self):
        """ returns q.conjugate()/q.norm()**2
        
        So if the quaternion is unit length, it is the same
        as the conjugate.
        """
        new = self.conjugate()
        tmp = self.norm()**2
        new.w /= tmp
        new.x /= tmp
        new.y /= tmp
        new.z /= tmp
        return new