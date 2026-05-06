def cree_ws_lecture(self, champs_ligne):
        """Alternative to create read only widgets. They should be set after."""
        for c in champs_ligne:
            label = ASSOCIATION[c][0]
            w = ASSOCIATION[c][3](self.acces[c], False)
            w.setObjectName("champ-lecture-seule-details")
            self.widgets[c] = (w, label)