def from_config(cls, pyvlx, item):
        """Read roller shutter from config."""
        name = item['name']
        ident = item['id']
        subtype = item['subtype']
        typeid = item['typeId']
        return cls(pyvlx, ident, name, subtype, typeid)