def forwards(self, orm):
        "Write your forwards methods here."
        for document in orm['document_library.Document'].objects.all():
            self.migrate_placeholder(
                orm, document, 'content', 'document_library_content', 'content')