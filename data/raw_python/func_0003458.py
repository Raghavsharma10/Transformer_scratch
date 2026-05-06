def links(self) -> _Links:
        """All found links on page, in as–is form.  Only works for Atom feeds."""
        return list(set(x.text for x in self.xpath('//link')))