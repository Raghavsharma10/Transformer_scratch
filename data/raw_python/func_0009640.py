def fromMarkdown(md, *args, **kwargs):
        """
        Creates abstraction using path to file

        :param str path: path to markdown file
        :return: TreeOfContents object
        """
        return TOC.fromHTML(markdown(md, *args, **kwargs))