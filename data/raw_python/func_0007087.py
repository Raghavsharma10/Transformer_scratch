def on_add(self, item):
        """Convert to pseuso acces"""
        super(Tels, self).on_add(list_views.PseudoAccesCategorie(item))