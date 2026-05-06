def format_raw_field(key):
    """
    When ElasticSearch analyzes string, it breaks it into parts.
    In order make query for not-analyzed exact string values, we should use subfield instead.

    The index template for Elasticsearch 5.0 has been changed.
    The subfield for string multi-fields has changed from .raw to .keyword

    Thus workaround for backward compatibility during migration is required.
    See also: https://github.com/elastic/logstash/blob/v5.4.1/docs/static/breaking-changes.asciidoc
    """
    subfield = django_settings.WALDUR_CORE.get('ELASTICSEARCH', {}).get('raw_subfield', 'keyword')
    return '%s.%s' % (camel_case_to_underscore(key), subfield)