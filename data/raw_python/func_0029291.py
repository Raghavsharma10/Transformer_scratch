def finalize(self):
        """Output the duplicate scripts detected."""
        if self.total_duplicate > 0:
            print('{} duplicate scripts found'.format(self.total_duplicate))
            for duplicate in self.list_duplicate:
                print(duplicate)