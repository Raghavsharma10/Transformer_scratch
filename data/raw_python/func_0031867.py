def fake_fk(self, field_name):
        """
        Return related random object to set as ForeignKey.

        Example Output:
            <User: username>
        """
        return self.djipsum_fields().getOrCreateForeignKey(
            model_class=self.model_class(),
            field_name=field_name
        )