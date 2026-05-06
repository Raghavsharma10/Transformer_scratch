def display_info(self):
        """Display basic information about the running MOC."""

        if self.moc is None:
            print('No MOC information present')
            return

        if self.moc.name is not None:
            print('Name:', self.moc.name)
        if self.moc.id is not None:
            print('Identifier:', self.moc.id)
        print('Order:', self.moc.order)
        print('Cells:', self.moc.cells)
        print('Area:', self.moc.area_sq_deg, 'square degrees')