def _clean_quantity(self, quantity):
        """Clean quantity value before it is added to entry."""
        value = quantity.get(QUANTITY.VALUE, '').strip()
        error = quantity.get(QUANTITY.E_VALUE, '').strip()
        unit = quantity.get(QUANTITY.U_VALUE, '').strip()
        kind = quantity.get(QUANTITY.KIND, '')

        if isinstance(kind, list) and not isinstance(kind, string_types):
            kind = [x.strip() for x in kind]
        else:
            kind = kind.strip()

        if not value:
            return False

        if is_number(value):
            value = '%g' % Decimal(value)
        if error:
            error = '%g' % Decimal(error)

        if value:
            quantity[QUANTITY.VALUE] = value
        if error:
            quantity[QUANTITY.E_VALUE] = error
        if unit:
            quantity[QUANTITY.U_VALUE] = unit
        if kind:
            quantity[QUANTITY.KIND] = kind

        return True