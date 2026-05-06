def _modulo(self, decimal_argument):
        """
        The mod operator is prone to floating point errors, so use decimal.

        101.1 % 100
        >>> 1.0999999999999943

        decimal_context.divmod(Decimal('100.1'), 100)
        >>> (Decimal('1'), Decimal('0.1'))
        """
        _times, remainder = self._context.divmod(decimal_argument, 100)

        # match the builtin % behavior by adding the N to the result if negative
        return remainder if remainder >= 0 else remainder + 100