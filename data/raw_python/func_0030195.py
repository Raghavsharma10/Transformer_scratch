def subclass(cls, vt_code, vt_args):
        """
        Return a dynamic subclass that has the extra parameters built in
        :param vt_code: The full VT code, privided to resolve_type
        :param vt_args: The portion of the VT code to the right of the part that matched a ValueType
        :return:
        """
        return type(vt_code.replace('/', '_'), (cls,), {'vt_code': vt_code, 'vt_args': vt_args})