def _pfp__show(self, level=0, include_offset=False):
        """Show the contents of the struct
        """
        res = []
        res.append("{}{} {{".format(
            "{:04x} ".format(self._pfp__offset) if include_offset else "",
            self._pfp__show_name
        ))
        for child in self._pfp__children:
            res.append("{}{}{:10s} = {}".format(
                "    "*(level+1),
                "{:04x} ".format(child._pfp__offset) if include_offset else "",
                child._pfp__name,
                child._pfp__show(level+1, include_offset)
            ))
        res.append("{}}}".format("    "*level))
        return "\n".join(res)