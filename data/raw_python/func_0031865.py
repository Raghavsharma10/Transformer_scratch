def fake_chars_or_choice(self, field_name):
        """
        Return fake chars or choice it if the `field_name` has choices.
        Then, returning random value from it.
        This specially for `CharField`.

        Usage:
            faker.fake_chars_or_choice('field_name')

        Example for field:
            TYPE_CHOICES = (
              ('project', 'I wanna to talk about project'),
              ('feedback', 'I want to report a bugs or give feedback'),
              ('hello', 'I just want to say hello')
            )
            type = models.CharField(max_length=200, choices=TYPE_CHOICES)
        """
        return self.djipsum_fields().randomCharField(
            self.model_class(),
            field_name=field_name
        )