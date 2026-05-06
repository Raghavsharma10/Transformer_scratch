def central_widget(self):
        """ Returns the default widget that occupies the entire area of the
        canvas. 
        """
        if self._central_widget is None:
            self._central_widget = Widget(size=self.size, parent=self.scene)
        return self._central_widget