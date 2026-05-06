def readable_size(C, bytes, suffix='B', decimals=1, sep='\u00a0'):
        """given a number of bytes, return the file size in readable units"""
        if bytes is None:
            return
        size = float(bytes)
        for unit in C.SIZE_UNITS:
            if abs(size) < 1024 or unit == C.SIZE_UNITS[-1]:
                return "{size:.{decimals}f}{sep}{unit}{suffix}".format(
                    size=size,
                    unit=unit,
                    suffix=suffix,
                    sep=sep,
                    decimals=C.SIZE_UNITS.index(unit) > 0 and decimals or 0,  # B with no decimal
                )
            size /= 1024