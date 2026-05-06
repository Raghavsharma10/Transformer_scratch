def wrap_setup(self, class_name, typ, with_async=False):
        """Described.typ = noy_wrap_typ(Described, Described.typ)"""
        equivalence = self.equivalence[typ]
        return [
              (NEWLINE, '\n')
            , (NAME, class_name)
            , (OP, '.')
            , (NAME, equivalence)
            , (OP, '=')
            , (NAME, "%snoy_wrap_%s" % ('async_' if with_async else '', equivalence))
            , (OP, "(")
            , (NAME, class_name)
            , (OP, ',')
            , (NAME, class_name)
            , (OP, '.')
            , (NAME, equivalence)
            , (OP, ")")
            ]