def get_parent(self, directory):
        """
        Given a directory name, return the Page representing it in the menu
        heirarchy.
        """
        assert settings.PAGE_DIR.startswith('/')
        assert settings.PAGE_DIR.endswith('/')

        parents = directory[len(settings.PAGE_DIR):]

        page = None
        if parents:
            for slug in parents.split('/'):
                page = Page.objects.get(parent=page, slug=slug)
        return page