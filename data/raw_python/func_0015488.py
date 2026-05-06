def schema(self, shex: Optional[Union[str, ShExJ.Schema]]) -> None:
        """ Set the schema to be used.  Schema can either be a ShExC or ShExJ string or a pre-parsed schema.

        :param shex:  Schema
        """
        self.pfx = None
        if shex is not None:
            if isinstance(shex, ShExJ.Schema):
                self._schema = shex
            else:
                shext = shex.strip()
                loader = SchemaLoader()
                if ('\n' in shex or '\r' in shex) or shext[0] in '#<_: ':
                    self._schema = loader.loads(shex)
                else:
                    self._schema = loader.load(shex) if isinstance(shex, str) else shex
                if self._schema is None:
                    raise ValueError("Unable to parse shex file")
                self.pfx = PrefixLibrary(loader.schema_text)