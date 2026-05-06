def fake_decimal(self, field_name):
        """
        Validate if the field has a `max_digits` and `decimal_places`
        And generating the unique decimal number.

        Usage:
            faker.fake_decimal('field_name')

        Example:
            10.7, 13041.00, 200.000.000
        """
        return self.djipsum_fields().randomDecimalField(
            self.model_class(),
            field_name=field_name
        )