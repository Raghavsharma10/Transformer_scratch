def origin_id_to_name(self, origin):
        """ Returns a localized origin name for a given ID """
        try:
            oid = int(origin)
        except (ValueError, TypeError):
            return None

        return self.origins.get(oid)