def _setup_delegate(self):
        """Add resize behavior on edit"""
        delegate = self.DELEGATE_CLASS(self)
        self.setItemDelegate(delegate)
        delegate.sizeHintChanged.connect(
            lambda index: self.resizeRowToContents(index.row()))
        if self.RESIZE_COLUMN:
            delegate.sizeHintChanged.connect(
                lambda index: self.resizeColumnToContents(index.column()))
        delegate.closeEditor.connect(
            lambda ed: self.resizeRowToContents(delegate.row_done_))