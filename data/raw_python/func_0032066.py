def productForm(self, request, tag):
        """
        Render a L{liveform.LiveForm} -- the main purpose of this fragment --
        which will allow the administrator to endow or deprive existing users
        using Products.
        """

        def makeRemover(i):
            def remover(s3lected):
                if s3lected:
                    return self.products[i]
                return None
            return remover

        f = liveform.LiveForm(
            self._endow,
            [liveform.Parameter(
                    'products' + str(i),
                    liveform.FORM_INPUT,
                    liveform.LiveForm(
                        makeRemover(i),
                        [liveform.Parameter(
                                's3lected',
                                liveform.RADIO_INPUT,
                                bool,
                                repr(p),
                                )],
                        '',
                        ),
                    )
             for (i, p)
             in enumerate(self.products)],
            self.which.capitalize() + u' ' + self.username)
        f.setFragmentParent(self)
        return f