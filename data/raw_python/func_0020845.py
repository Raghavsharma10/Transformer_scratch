def _schema_from_verb(verb, partial=False):
    """Return an instance of schema for given verb."""
    from .verbs import Verbs
    return getattr(Verbs, verb)(partial=partial)