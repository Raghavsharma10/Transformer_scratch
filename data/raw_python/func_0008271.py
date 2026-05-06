def is_abstract(self) -> bool:
        """
        Whether or not the class-under-construction was declared as abstract (**NOTE:**
        this property is usable even *before* the :class:`MetaOptionsFactory` has run)
        """
        meta_value = getattr(self.clsdict.get('Meta'), 'abstract', False)
        return self.clsdict.get(ABSTRACT_ATTR, meta_value) is True