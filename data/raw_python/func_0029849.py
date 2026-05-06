def get_object(self):
        """Return contents in object form, an AttrDict"""

        from ..util import AttrDict

        c = self.record.unpacked_contents

        if not c:
            c = yaml.safe_load(self.default)

        return AttrDict(c)