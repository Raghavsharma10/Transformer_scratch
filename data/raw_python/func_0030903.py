def compute(a, b, axis):
        """
        Finds optimal displacements localized along an axis
        """
        delta = []
        for aa, bb in zip(rollaxis(a, axis, 0), rollaxis(b, axis, 0)):
            delta.append(Displacement.compute(aa, bb).delta)
        return LocalDisplacement(delta, axis=axis)