def order_by(self, *fields):
        """An alternate to ``sort`` which allows you to specify a list
        of fields and use a leading - (minus) to specify DESCENDING."""
        doc = []
        for field in fields:
            if field.startswith('-'):
                doc.append((field.strip('-'), pymongo.DESCENDING))
            else:
                doc.append((field, pymongo.ASCENDING))
        return self.sort(doc)