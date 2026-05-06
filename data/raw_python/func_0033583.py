def forwards(self, orm):
        "Write your forwards methods here."
        for category in orm['document_library.DocumentCategory'].objects.all():
            for trans_old in orm['document_library.DocumentCategoryTitle'].objects.filter(category=category):
                orm['document_library.DocumentCategoryTranslation'].objects.create(
                    master=category,
                    language_code=trans_old.language,
                    title=trans_old.title,
                )

        for document in orm['document_library.Document'].objects.all():
            for trans_old in orm['document_library.DocumentTitle'].objects.filter(document=document):
                orm['document_library.DocumentTranslation'].objects.create(
                    master=document,
                    language_code=trans_old.language,
                    title=trans_old.title,
                    description=trans_old.description,
                    filer_file=trans_old.filer_file,
                    thumbnail=trans_old.thumbnail,
                    copyright_notice=trans_old.copyright_notice,
                    is_published=trans_old.is_published,
                    meta_description=trans_old.meta_description,
                )