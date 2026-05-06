def next_val(self, field):
        """Return a new value to mutate a field with. Do not modify the field directly
        in this function. Override the ``mutate()`` function if that is needed (the field is
        only passed into this function as a reference).

        :field: The pfp.fields.Field instance that will receive the new value. Passed in for reference only.
        :returns: The next value for the field
        """
        import pfp.fuzz.rand as rand

        if self.choices is not None:
            choices = self._resolve_member_val(self.choices, field)
            new_val = rand.choice(choices)
            return self._resolve_val(new_val)

        elif self.prob is not None:
            prob = self._resolve_member_val(self.prob, field)
            rand_val = rand.random()
            curr_total = 0.0
            # iterate through each of the probability choices until
            # we reach one that matches the current rand_val
            for prob_percent, prob_val in prob:
                if rand_val <= curr_total + prob_percent:
                    return self._resolve_val(prob_val)
                curr_total += prob_percent

            raise MutationError("probabilities did not add up to 100%! {}".format(
                [str(x[0]) + " - " + str(x[1])[:10] for x in prob]
            ))