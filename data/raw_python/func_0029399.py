def finalize(self):
        """Output the default sprite names found in the project."""
        print('{} default sprite names found:'.format(self.total_default))
        for name in self.list_default:
            print(name)