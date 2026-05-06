def make_header(self, locale, catalog):
        """Populate header with correct data from top-most locale file."""
        return {
            "po-revision-date": self.get_catalogue_header_value(catalog, 'PO-Revision-Date'),
            "mime-version": self.get_catalogue_header_value(catalog, 'MIME-Version'),
            "last-translator": 'Automatic <hi@thorgate.eu>',
            "x-generator": "Python",
            "language": self.get_catalogue_header_value(catalog, 'Language') or locale,
            "lang": locale,
            "content-transfer-encoding": self.get_catalogue_header_value(catalog, 'Content-Transfer-Encoding'),
            "project-id-version": self.get_catalogue_header_value(catalog, 'Project-Id-Version'),
            "pot-creation-date": self.get_catalogue_header_value(catalog, 'POT-Creation-Date'),
            "domain": self.domain,
            "report-msgid-bugs-to": self.get_catalogue_header_value(catalog, 'Report-Msgid-Bugs-To'),
            "content-type": self.get_catalogue_header_value(catalog, 'Content-Type'),
            "plural-forms": self.get_plural(catalog),
            "language-team": self.get_catalogue_header_value(catalog, 'Language-Team')
        }