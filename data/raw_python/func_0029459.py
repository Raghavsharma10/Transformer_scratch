def one_symbol_ops_str(self) -> str:
        """Regex-escaped string with all one-symbol operators"""
        return re.escape(''.join((key for key in self.ops.keys() if len(key) == 1)))