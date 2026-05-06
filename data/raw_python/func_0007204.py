def _draw_placeholder(self):
        """To be used in QTreeView"""
        if self.model().rowCount() == 0:
            painter = QPainter(self.viewport())
            painter.setFont(_custom_font(is_italic=True))
            painter.drawText(self.rect().adjusted(0, 0, -5, -5), Qt.AlignCenter | Qt.TextWordWrap,
                             self.PLACEHOLDER)