def add_layout(self, l):
        """Convenience function"""
        if self.layout():
            self.layout().addLayout(l)
        else:
            layout = QVBoxLayout(self)
            layout.addLayout(l)