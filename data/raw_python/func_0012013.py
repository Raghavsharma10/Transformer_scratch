def save(self, *args, **kwargs):
        """
        Extends save() method of Django models to check that the database name
        is not left blank.
        Note: 'blank=False' is only checked at a form-validation-stage. A test
        using Fixtureless that tries to randomly create a CrossRefDB with an
        empty string name would unintentionally break the test.
        """
        if self.name == '':
            raise FieldError
        else:
            return super(CrossRefDB, self).save(*args, **kwargs)