def save_related(self, request, form, formsets, change):
        """
        Rebuilds the tree after saving items related to parent.
        """
        super(MenuItemAdmin, self).save_related(request, form, formsets, change)
        self.model.objects.rebuild()