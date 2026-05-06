def _clean_code(self, code):
        """ Return *code* with indentation and leading/trailing blank lines
        removed. 
        """
        lines = code.split("\n")
        min_indent = 100
        for line in lines:
            if line.strip() != "":
                indent = len(line) - len(line.lstrip())
                min_indent = min(indent, min_indent)
        if min_indent > 0:
            lines = [line[min_indent:] for line in lines]
        code = "\n".join(lines)
        return code