def add(self, num):
        """
        Adds num to the current value, jumping up the next
        multiple of mfac if the result is not a multiple already
        """
        try:
            val = self.value() + num
        except:
            val = num

        chunk = self.mfac.value()
        if val % chunk > 0:
            if num > 0:
                val = chunk*(val // chunk + 1)
            elif num < 0:
                val = chunk*(val // chunk)

        val = max(self._min(), min(self._max(), val))
        self.set(val)