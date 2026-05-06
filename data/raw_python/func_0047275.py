def sub_article_folders(self):
        """
        Returns all valid ArticleFolder sitting inside of
        :attr:`ArticleFolder.dir_path`.
        """
        l = list()
        for p in Path.sort_by_fname(
                Path(self.dir_path).select_dir(recursive=False)
        ):
            af = ArticleFolder(dir_path=p.abspath)
            try:
                if af.title is not None:
                    l.append(af)
            except:
                pass
        return l