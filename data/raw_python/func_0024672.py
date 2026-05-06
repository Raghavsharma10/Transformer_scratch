def _do_setup(self):
        """Setup basic parameters for this class.

`base` is the numeric base which when raised to `power` is equivalent
to 1 unit of the corresponding prefix. I.e., base=2, power=10
represents 2^10, which is the NIST Binary Prefix for 1 Kibibyte.

Likewise, for the SI prefix classes `base` will be 10, and the `power`
for the Kilobyte is 3.
"""
        (self._base, self._power, self._name_singular, self._name_plural) = self._setup()
        self._unit_value = self._base ** self._power