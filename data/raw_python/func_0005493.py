def add_taxes(self, taxes):
        """Appends the data to the 'taxes' key in the request object

        'taxes' should be in format: [("tax_name", "tax_amount")]
        For example:
        [("Other TAX", 700), ("VAT", 5000)]
        """
        # fixme: how to resolve duplicate tax names
        _idx = len(self.taxes)  # current index to prevent overwriting
        for idx, tax in enumerate(taxes):
            tax_key = "tax_" + str(idx + _idx)
            self.taxes[tax_key] = {"name": tax[0], "amount": tax[1]}