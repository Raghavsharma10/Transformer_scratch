def fly(cls,
            conf_path,
            docname,
            source,
            maxdepth=1):  # pragma: no cover
        """
        Generate toctree directive for rst file.

        :param conf_path: conf.py file absolute path
        :param docname: the rst file relpath from conf.py directory.
        :param source: rst content.
        :param maxdepth: int, max toc tree depth.
        """
        msg = ("``.. articles::`` directive is going to be deprecated. "
               "use ``.. autodoctree`` instead.")
        warnings.warn(msg, FutureWarning)

        directive_pattern = ".. articles::"
        if directive_pattern not in source:
            return source

        af = ArticleFolder(
            dir_path=Path(Path(conf_path).parent, docname).parent.abspath)
        toc_directive = af.toc_directive(maxdepth)

        lines = list()
        for line in source.split("\n"):
            if directive_pattern in line.strip():
                if line.strip().startswith(directive_pattern):
                    line = line.replace(directive_pattern, toc_directive, 1)
                    lines.append(line)
                    continue
            lines.append(line)
        return "\n".join(lines)