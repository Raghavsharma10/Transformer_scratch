def derive_toctree_rst(self, current_file):
        """
        Generate the rst content::

            .. toctree::
                args ...

                example.rst
                ...

        :param current_file:
        :return:
        """
        TAB = " " * 4
        lines = list()
        lines.append(".. toctree::")
        for opt in TocTree.option_spec:
            value = self.options.get(opt)
            if value is not None:
                lines.append(("{}:{}: {}".format(TAB, opt, value)).rstrip())
        lines.append("")

        append_ahead = "append_ahead" in self.options
        if append_ahead:
            for line in list(self.content):
                lines.append(TAB + line)

        article_folder = ArticleFolder(dir_path=Path(current_file).parent.abspath)
        for af in article_folder.sub_article_folders:
            line = "{}{} <{}>".format(TAB, af.title, af.rel_path)
            lines.append(line)

        append_behind = not append_ahead
        if append_behind:
            for line in list(self.content):
                lines.append(TAB + line)

        lines.append("")
        return "\n".join(lines)