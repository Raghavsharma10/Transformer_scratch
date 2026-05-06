def _fake_property(self, p, next, indexed=True):
    """Internal helper to create a fake Property."""
    self._clone_properties()
    if p.name() != next and not p.name().endswith('.' + next):
      prop = StructuredProperty(Expando, next)
      prop._store_value(self, _BaseValue(Expando()))
    else:
      compressed = p.meaning_uri() == _MEANING_URI_COMPRESSED
      prop = GenericProperty(next,
                             repeated=p.multiple(),
                             indexed=indexed,
                             compressed=compressed)
    prop._code_name = next
    self._properties[prop._name] = prop
    return prop