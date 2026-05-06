def post_save(self):
        """can be removed when a proper task manager admin interface implemented"""
        if self.run:
            self.run = False
            self.create_tasks()
            self.save()