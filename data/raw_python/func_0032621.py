def beforeRender(self, ctx):
        """
        Call the C{beforeRender} implementations on L{MantissaLivePage} and
        L{_FragmentWrapperMixin}.
        """
        MantissaLivePage.beforeRender(self, ctx)
        return _FragmentWrapperMixin.beforeRender(self, ctx)