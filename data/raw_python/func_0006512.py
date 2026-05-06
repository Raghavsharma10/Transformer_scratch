def add_widget(self, w):
        """Convenience function"""
        if self.layout():
            self.layout().addWidget(w)
        else:
            layout = QVBoxLayout(self)
            layout.addWidget(w)