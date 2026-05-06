def forwards(self, orm):
        "Write your forwards methods here."
        # Note: Remember to use orm['appname.ModelName'] rather than "from appname.models..."
        for entry in orm['multilingual_news.NewsEntry'].objects.all():
            self.migrate_placeholder(
                orm, entry, 'excerpt', 'multilingual_news_excerpt', 'excerpt')
            self.migrate_placeholder(
                orm, entry, 'content', 'multilingual_news_content', 'content')