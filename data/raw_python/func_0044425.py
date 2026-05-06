def _mode(self, s):
        """Check file mode format and parse into an int.

        :return: mode as integer
        """
        # Note: Output from git-fast-export slightly different to spec
        if s in [b'644', b'100644', b'0100644']:
            return 0o100644
        elif s in [b'755', b'100755', b'0100755']:
            return 0o100755
        elif s in [b'040000', b'0040000']:
            return 0o40000
        elif s in [b'120000', b'0120000']:
            return 0o120000
        elif s in [b'160000', b'0160000']:
            return 0o160000
        else:
            self.abort(errors.BadFormat, 'filemodify', 'mode', s)