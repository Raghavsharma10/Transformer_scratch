def get_text(self, node):
        """Get node text representation."""
        return click.style(
            repr(node), fg='green' if node.level > 1 else 'red'
        )