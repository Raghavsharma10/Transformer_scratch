def backwards(self, orm):
        "Write your backwards methods here."
        orm['avocado.DataConcept'].objects.filter(name='Sample')\
                .update(queryable=True)