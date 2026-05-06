def bulk_index_relations(cls, items, request=None, **kwargs):
        """ Index objects related to :items: in bulk.

        Related items are first grouped in map
        {model_name: {item1, item2, ...}} and then indexed.

        :param items: Sequence of DB objects related objects if which
            should be indexed.
        :param request: Pyramid Request instance.
        """
        index_map = defaultdict(set)
        for item in items:
            relations = item.get_related_documents(**kwargs)
            for model_cls, related_items in relations:
                indexable = getattr(model_cls, '_index_enabled', False)
                if indexable and related_items:
                    index_map[model_cls.__name__].update(related_items)

        for model_name, instances in index_map.items():
            cls(model_name).index(to_dicts(instances), request=request)