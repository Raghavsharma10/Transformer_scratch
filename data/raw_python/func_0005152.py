def _format_numeric_sequence(self, _sequence, separator="."):
        """ Length of the highest index in chars = justification size """
        if not _sequence:
            return colorize(_sequence, "purple")
        _sequence = _sequence if _sequence is not None else self.obj
        minus = (2 if self._depth > 0 else 0)
        just_size = len(str(len(_sequence)))
        out = []
        add_out = out.append
        for i, item in enumerate(_sequence):
            self._incr_just_size(just_size+minus)
            add_out(self._numeric_prefix(
                i, self.pretty(item, display=False),
                just=just_size, color="blue", separator=separator))
            self._decr_just_size(just_size+minus)
        if not self._depth:
            return padd("\n".join(out) if out else str(out), padding="top")
        else:
            return "\n".join(out) if out else str(out)