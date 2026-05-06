def _check_endings(self):
        """Check begin/end of slug, raises Error if malformed."""
        if self.slug.startswith("/") and self.slug.endswith("/"):
            raise InvalidSlugError(
                _("Invalid slug. Did you mean {}, without the leading and trailing slashes?".format(self.slug.strip("/"))))
        elif self.slug.startswith("/"):
            raise InvalidSlugError(
                _("Invalid slug. Did you mean {}, without the leading slash?".format(self.slug.strip("/"))))
        elif self.slug.endswith("/"):
            raise InvalidSlugError(
                _("Invalid slug. Did you mean {}, without the trailing slash?".format(self.slug.strip("/"))))