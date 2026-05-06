def add_block_widget(self, top=False):
        """
        Return a select widget for blocks which can be added to this column.
        """
        widget = AddBlockSelect(attrs={
            'class': 'glitter-add-block-select',
        }, choices=self.add_block_options(top=top))

        return widget.render(name='', value=None)