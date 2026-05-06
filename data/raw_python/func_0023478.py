def get_breakpoints(self, lineno):
        """Return the list of breakpoints set at lineno."""
        try:
            firstlineno, actual_lno = self.bdb_module.get_actual_bp(lineno)
        except BdbSourceError:
            return []
        if firstlineno not in self:
            return []
        code_bps = self[firstlineno]
        if actual_lno not in code_bps:
            return []
        return [bp for bp in sorted(code_bps[actual_lno],
                    key=attrgetter('number')) if bp.line == lineno]