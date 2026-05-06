def forwards(self, orm):
        "Write your forwards methods here."
        # Note: Remember to use orm['appname.ModelName'] rather than "from appname.models..."
        for entry_title in orm.NewsEntryTitle.objects.all():
            entry = NewsEntry.objects.get(pk=entry_title.entry.pk)
            entry.translate(entry_title.language)
            entry.title = entry_title.title
            entry.slug = entry_title.slug
            entry.is_published = entry_title.is_published
            entry.save()