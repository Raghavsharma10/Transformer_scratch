def clean_publish_dates(self):
        """
        If an end_date value is provided, the start_date must be less.
        """
        if self.end_date:
            if not self.start_date:
                raise ValidationError("""The End Date requires a Start Date value.""")
            elif self.end_date <= self.start_date:
                raise ValidationError("""The End Date must not precede the Start Date.""")