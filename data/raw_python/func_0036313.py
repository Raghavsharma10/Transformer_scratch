def create_manager(self, instance, superclass):
        """
        Dynamically create a RelatedManager to handle the back side of the (G)FK
        """
        rel_model = self.rating_model
        rated_model = self.rated_model

        class RelatedManager(superclass):
            def get_query_set(self):
                qs = RatingsQuerySet(rel_model, rated_model=rated_model)
                return qs.filter(**(self.core_filters))

            def add(self, *objs):
                lookup_kwargs = rel_model.lookup_kwargs(instance)
                for obj in objs:
                    if not isinstance(obj, self.model):
                        raise TypeError("'%s' instance expected" %
                                        self.model._meta.object_name)
                    for (k, v) in lookup_kwargs.iteritems():
                        setattr(obj, k, v)
                    obj.save()
            add.alters_data = True

            def create(self, **kwargs):
                kwargs.update(rel_model.lookup_kwargs(instance))
                return super(RelatedManager, self).create(**kwargs)
            create.alters_data = True

            def get_or_create(self, **kwargs):
                kwargs.update(rel_model.lookup_kwargs(instance))
                return super(RelatedManager, self).get_or_create(**kwargs)
            get_or_create.alters_data = True

            def remove(self, *objs):
                for obj in objs:
                    # Is obj actually part of this descriptor set?
                    if obj in self.all():
                        obj.delete()
                    else:
                        raise rel_model.DoesNotExist(
                            "%r is not related to %r." % (obj, instance))
            remove.alters_data = True

            def clear(self):
                self.all().delete()
            clear.alters_data = True

            def rate(self, user, score):
                rating, created = self.get_or_create(user=user)
                if created or score != rating.score:
                    rating.score = score
                    rating.save()
                return rating

            def unrate(self, user):
                return self.filter(user=user,
                                   **rel_model.lookup_kwargs(instance)
                                   ).delete()

            def perform_aggregation(self, aggregator):
                score = self.all().aggregate(agg=aggregator('score'))
                return score['agg']

            def cumulative_score(self):
                # simply the sum of all scores, useful for +1/-1
                return self.perform_aggregation(models.Sum)

            def average_score(self):
                # the average of all the scores, useful for 1-5
                return self.perform_aggregation(models.Avg)

            def standard_deviation(self):
                # the standard deviation of all the scores, useful for 1-5
                return self.perform_aggregation(models.StdDev)

            def variance(self):
                # the variance of all the scores, useful for 1-5
                return self.perform_aggregation(models.Variance)

            def similar_items(self):
                return SimilarItem.objects.get_for_item(instance)

        manager = RelatedManager()
        manager.core_filters = rel_model.lookup_kwargs(instance)
        manager.model = rel_model

        return manager