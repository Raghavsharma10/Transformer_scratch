def can_regex(self, field):
        '''Test if a given field supports regex lookups'''
        from django.conf import settings
        if settings.DATABASES['default']['ENGINE'].endswith('sqlite3'):
            return not isinstance(get_real_field(self.model, field), UNSUPPORTED_REGEX_FIELDS)
        else:
            return True