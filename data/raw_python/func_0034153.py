def get_changelog_file_for_database(database=DEFAULT_DB_ALIAS):
    """get changelog filename for given `database` DB alias"""

    from django.conf import settings

    try:
        return settings.LIQUIMIGRATE_CHANGELOG_FILES[database]
    except AttributeError:
        if database == DEFAULT_DB_ALIAS:
            try:
                return settings.LIQUIMIGRATE_CHANGELOG_FILE
            except AttributeError:
                raise ImproperlyConfigured(
                        'Please set LIQUIMIGRATE_CHANGELOG_FILE or '
                        'LIQUIMIGRATE_CHANGELOG_FILES in your '
                        'project settings')
        else:
            raise ImproperlyConfigured(
                'LIQUIMIGRATE_CHANGELOG_FILES dictionary setting '
                'is required for multiple databases support')
    except KeyError:
        raise ImproperlyConfigured(
            "Liquibase changelog file is not set for database: %s" % database)