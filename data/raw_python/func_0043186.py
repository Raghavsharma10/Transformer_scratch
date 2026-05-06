def destroy(self, configuration=None):
        """
        Completely destroys a store and all the contexts and triples in the store.

        >>> store = DjangoStore()
        >>> g = rdflib.Graph(store=store)
        >>> g.open(configuration=None, create=True) == rdflib.store.VALID_STORE
        True
        >>> g.open(configuration=None, create=False) == rdflib.store.VALID_STORE
        True
        >>> g.destroy(configuration=None)
        >>> g.open(configuration=None, create=False) == rdflib.store.VALID_STORE
        True
        """
        models.NamedGraph.objects.all().delete()
        models.URIStatement.objects.all().delete()
        models.LiteralStatement.objects.all().delete()