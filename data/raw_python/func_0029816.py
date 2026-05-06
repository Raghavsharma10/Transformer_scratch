def _jinja_sub(self, st):
        """Create a Jina template engine, then perform substitutions on a string"""

        if isinstance(st, string_types):
            from jinja2 import Template

            try:
                for i in range(5):  # Only do 5 recursive substitutions.
                    st = Template(st).render(**(self._top.dict))
                    if '{{' not in st:
                        break
                return st
            except Exception as e:
                return st
                #raise ValueError(
                #    "Failed to render jinja template for metadata value '{}': {}".format(st, e))

        return st