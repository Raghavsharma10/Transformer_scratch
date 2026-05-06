def get_content_title(self, obj):
        """Get content's title."""
        return Content.objects.get(id=obj.content.id).title