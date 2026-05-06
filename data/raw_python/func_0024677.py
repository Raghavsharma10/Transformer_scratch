def format(self, fmt):
        """Return a representation of this instance formatted with user
supplied syntax"""
        _fmt_params = {
            'base': self.base,
            'bin': self.bin,
            'binary': self.binary,
            'bits': self.bits,
            'bytes': self.bytes,
            'power': self.power,
            'system': self.system,
            'unit': self.unit,
            'unit_plural': self.unit_plural,
            'unit_singular': self.unit_singular,
            'value': self.value
        }

        return fmt.format(**_fmt_params)