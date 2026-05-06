def add_quantity(self,
                     quantities,
                     value,
                     source,
                     check_for_dupes=True,
                     compare_to_existing=True,
                     **kwargs):
        """Add an `Quantity` instance to this entry."""
        success = True
        for quantity in listify(quantities):
            kwargs.update({QUANTITY.VALUE: value, QUANTITY.SOURCE: source})
            cat_dict = self._add_cat_dict(
                Quantity,
                quantity,
                compare_to_existing=compare_to_existing,
                check_for_dupes=check_for_dupes,
                **kwargs)
            if isinstance(cat_dict, CatDict):
                self._append_additional_tags(quantity, source, cat_dict)
                success = False

        return success