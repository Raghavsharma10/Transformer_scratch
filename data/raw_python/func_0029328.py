def finalize(self):
        """Output the aggregate block count results."""
        for name, count in sorted(self.blocks.items(), key=lambda x: x[1]):
            print('{:3} {}'.format(count, name))
        print('{:3} total'.format(sum(self.blocks.values())))