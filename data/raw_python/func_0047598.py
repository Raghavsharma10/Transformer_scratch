def _tree_view_builder(self, indent=0, is_root=True):
        """
        Build a text to represent the package structure.
        """

        def pad_text(indent):
            return "    " * indent + "|-- "

        lines = list()

        if is_root:
            lines.append(SP_DIR)

        lines.append(
            "%s%s (%s)" % (pad_text(indent), self.shortname, self.fullname)
        )

        indent += 1

        # sub packages
        for pkg in self.sub_packages.values():
            lines.append(pkg._tree_view_builder(indent=indent, is_root=False))

        # __init__.py
        lines.append(
            "%s%s (%s)" % (
                pad_text(indent), "__init__.py", self.fullname,
            )
        )

        # sub modules
        for mod in self.sub_modules.values():
            lines.append(
                "%s%s (%s)" % (
                    pad_text(indent), mod.shortname + ".py", mod.fullname,
                )
            )

        return "\n".join(lines)