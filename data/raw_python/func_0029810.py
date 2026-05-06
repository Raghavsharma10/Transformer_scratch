def rows(self):
        """Return configuration in a form that can be used to reconstitute a
        Metadata object. Returns all of the rows for a dataset.

        This is distinct from get_config_value, which returns the value
        for the library.

        """
        from ambry.orm import Config as SAConfig
        from sqlalchemy import or_

        rows = []
        configs = self.dataset.session\
            .query(SAConfig)\
            .filter(or_(SAConfig.group == 'config', SAConfig.group == 'process'),
                    SAConfig.d_vid == self.dataset.vid)\
            .all()

        for r in configs:
            parts = r.key.split('.', 3)

            if r.group == 'process':
                parts = ['process'] + parts

            cr = ((parts[0] if len(parts) > 0 else None,
                   parts[1] if len(parts) > 1 else None,
                   parts[2] if len(parts) > 2 else None
                   ), r.value)

            rows.append(cr)

        return rows