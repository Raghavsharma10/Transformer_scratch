def doesIntersect(self, other):
        '''
        :param: other - Triangle or Line subclass
        :return: boolean

        Returns True iff:
           Any segment in self intersects any segment in other.

        '''
        otherType = type(other)

        if issubclass(otherType, Triangle):
            for s in self.segments.values():
                for q in other.segments.values():
                    if s.doesIntersect(q):
                        return True
            return False

        if issubclass(otherType, Line):
            for s in self.segments.values():
                if s.doesIntersect(other):
                    return True
            return False

        msg = "expecting Line or Triangle subclasses, got '{}'"

        raise TypeError(msg.format(otherType))