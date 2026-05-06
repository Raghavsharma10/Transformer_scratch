def forwards(self, orm):
        "Write your forwards methods here."
        # Note: Remember to use orm['appname.ModelName'] rather than "from appname.models..."

        User = orm[user_orm_label]

        try:
            user = User.objects.all()[0]

            for article in orm.Article.objects.all():
                article.author = user
                article.save()

        except IndexError:
            pass