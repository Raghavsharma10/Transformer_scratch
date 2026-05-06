def forwards(self, orm):
        "Write your forwards methods here."
        orm['avocado.DataConcept'].objects.filter(name='Sample')\
                .update(queryable=False)