def randomDecimalField(self, model_class, field_name):
        """
        Validate if the field has a `max_digits` and `decimal_places`
        And generating the unique decimal number.
        """
        decimal_field = model_class._meta.get_field(field_name)
        max_digits = None
        decimal_places = None

        if decimal_field.max_digits is not None:
            max_digits = decimal_field.max_digits
        if decimal_field.decimal_places is not None:
            decimal_places = decimal_field.decimal_places

        digits = random.choice(range(100))
        if max_digits is not None:
            start = 0
            if max_digits < start:
                start = max_digits - max_digits
            digits = int(
                "".join([
                    str(x) for x in random.sample(
                        range(start, max_digits),
                        max_digits - 1
                    )
                ])
            )
        places = random.choice(range(10, 99))
        if decimal_places is not None:
            places = str(
                random.choice(range(9999 * 99999))
            )[:decimal_places]

        return float(
            str(digits)[:decimal_places] + "." + str(places)
        )