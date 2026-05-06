def forwards(self, orm):
        "Write your forwards methods here."
        # Note: Remember to use orm['appname.ModelName'] rather than "from appname.models..."
        for category in orm['document_library.DocumentCategory'].objects.all():
            category.slug = category.documentcategorytitle_set.all()[0].title.lower()
            category.save()