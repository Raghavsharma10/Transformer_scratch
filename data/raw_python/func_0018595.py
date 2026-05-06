def __regkey_value(self, path, name='', start_key=None):
        r'''Return the data of value mecabrc at MeCab HKEY node.

        On Windows, the path to the mecabrc as set in the Windows Registry is
        used to deduce the path to libmecab.dll.

        Returns:
            The full path to the mecabrc on Windows.

        Raises:
            WindowsError: A problem was encountered in trying to locate the
                value mecabrc at HKEY_CURRENT_USER\Software\MeCab.
        '''
        if sys.version < '3':
            import _winreg as reg
        else:
            import winreg as reg

        def _fn(path, name='', start_key=None):
            if isinstance(path, str):
                path = path.split('\\')
            if start_key is None:
                start_key = getattr(reg, path[0])
                return _fn(path[1:], name, start_key)
            else:
                subkey = path.pop(0)
            with reg.OpenKey(start_key, subkey) as handle:
                if path:
                    return _fn(path, name, handle)
                else:
                    desc, i = None, 0
                    while not desc or desc[0] != name:
                        desc = reg.EnumValue(handle, i)
                        i += 1
                    return desc[1]
        return _fn(path, name, start_key)