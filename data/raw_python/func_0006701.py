def _set_boutons_interface(self, buttons):
        """Display buttons given by the list of tuples (id,function,description,is_active)"""
        for id_action, f, d, is_active in buttons:
            icon = self.get_icon(id_action)
            action = self.addAction(QIcon(icon), d)
            action.setEnabled(is_active)
            action.triggered.connect(f)