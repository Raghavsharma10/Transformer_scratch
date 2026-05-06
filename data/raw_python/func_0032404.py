def simulate(self):
        """Returns a randomly constructed string.

        Simulate randomly constructs a string with a length between min and
        max. If min is not present, a minimum length of 1 is assumed, if max
        is not present a maximum length of 10 is used.
        """
        min_ = 1 if self._min is None else self._min
        max_ = 10 if self._max is None else self._max
        n = min_ if (min_ >= max_) else random.randint(min_, max_)
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for x in range(n))