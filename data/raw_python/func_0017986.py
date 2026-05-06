def applyTuple(self, tuple, right, env):
    """Apply a tuple to something else."""
    if len(right) != 1:
      raise exceptions.EvaluationError('Tuple (%r) can only be applied to one argument, got %r' % (self.left, self.right))
    right = right[0]

    return tuple(right)