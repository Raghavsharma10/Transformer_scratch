def get_links(self, **kw):
        """
        Prepare links of form by mimicing pyoko's get_links method's result

        Args:
            **kw:

        Returns: list of link dicts

        """

        links = [a for a in dir(self) if isinstance(getattr(self, a), Model)
                 and not a.startswith('_model')]

        return [
            {
                'field': l,
                'mdl': getattr(self, l).__class__,
            } for l in links
        ]