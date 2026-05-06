def _mutate(self, condition, situation):
        """Create a new condition from the given one by probabilistically
        applying point-wise mutations. Bits that were originally wildcarded
        in the parent condition acquire their values from the provided
        situation, to ensure the child condition continues to match it."""

        # Go through each position in the condition, randomly flipping
        # whether the position is a value (0 or 1) or a wildcard (#). We do
        # this in a new list because the original condition's mask is
        # immutable.
        mutation_points = bitstrings.BitString.random(
            len(condition.mask),
            self.mutation_probability
        )
        mask = condition.mask ^ mutation_points

        # The bits that aren't wildcards always have the same value as the
        # situation, which ensures that the mutated condition still matches
        # the situation.
        if isinstance(situation, bitstrings.BitCondition):
            mask &= situation.mask
            return bitstrings.BitCondition(situation.bits, mask)
        return bitstrings.BitCondition(situation, mask)