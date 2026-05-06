def save(self, *args, **kwargs):
        '''The slug field is auto-populated during the save from the name field.'''
        if not self.id:
            self.slug = slugify(self.name)
        super(Cohort, self).save(*args, **kwargs)