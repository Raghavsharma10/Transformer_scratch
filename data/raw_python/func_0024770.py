def from_config(cls, pyvlx, item):
        """Read scene from configuration."""
        name = item['name']
        ident = item['id']
        return cls(pyvlx, ident, name)