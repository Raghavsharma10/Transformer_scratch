def filter(self, value=None, exclude=False, **selection):
        """ Most of the actual functionality lives on the Column
        object and the `all` and `any` functions. """
        filters = self.meta.setdefault('filters', [])

        if value and len(selection):
            raise ValueError("Cannot specify a filter string and a filter keyword selection at the same time.")
        elif value:
            value = [value]
        elif len(selection):
            value = select(self.api.columns, selection, invert=exclude)

        filters.append(value)
        self.raw['filters'] = utils.paste(filters, ',', ';')
        return self