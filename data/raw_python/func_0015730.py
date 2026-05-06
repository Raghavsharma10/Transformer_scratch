def _fixup_cdef_enums(string, reg=re.compile(r"=\s*(\d+)\s*<<\s*(\d+)")):
    """Converts some common enum expressions to constants"""

    def repl_shift(match):
        shift_by = int(match.group(2))
        value = int(match.group(1))
        int_value = ctypes.c_int(value << shift_by).value
        return "= %s" % str(int_value)
    return reg.sub(repl_shift, string)