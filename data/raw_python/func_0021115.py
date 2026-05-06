def has_urls(self):
        "Handy for templates."
        if self.isbn_uk or self.isbn_us or self.official_url or self.notes_url:
            return True
        else:
            return False