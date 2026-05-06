def track(cls, obj, ptr):
        """
        Track an object which needs destruction when it is garbage collected.
        """
        cls._objects.add(cls(obj, ptr))