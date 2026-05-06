def forwards(self, orm):
        "Write your forwards methods here."
        for doc in orm['document_library.Document'].objects.all():
            for title in doc.documenttitle_set.all():
                title.is_published = doc.is_published
                title.save()