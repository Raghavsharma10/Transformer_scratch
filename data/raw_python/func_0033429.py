def forwards(self, orm):
        "Write your forwards methods here."
        for category in orm['document_library.DocumentCategory'].objects.all():
            category.is_published = True
            category.save()