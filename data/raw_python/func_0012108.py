def bytes_from_readable_size(C, size, suffix='B'):
        """given a readable_size (as produced by File.readable_size()), return the number of bytes."""
        s = re.split("^([0-9\.]+)\s*([%s]?)%s?" % (''.join(C.SIZE_UNITS), suffix), size, flags=re.I)
        bytes, unit = round(float(s[1])), s[2].upper()
        while unit in C.SIZE_UNITS and C.SIZE_UNITS.index(unit) > 0:
            bytes *= 1024
            unit = C.SIZE_UNITS[C.SIZE_UNITS.index(unit) - 1]
        return bytes