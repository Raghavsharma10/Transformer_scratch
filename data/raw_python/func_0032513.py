def realForm(self, req, tag):
        """
        Render L{liveForm}.
        """
        self.liveForm.setFragmentParent(self)
        return self.liveForm