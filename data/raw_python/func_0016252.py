def from_search_query(self, search_query):
        """
        Return queryset of objects from SearchQuery.results, **in order**.

        EXPERIMENTAL: this will only work with results from a single index,
        with a single doc_type - as we are returning a single QuerySet.

        This method takes the hits JSON and converts that into a queryset
        of all the relevant objects. The key part of this is the ordering -
        the order in which search results are returned is based on relevance,
        something that only ES can calculate, and that cannot be replicated
        in the database.

        It does this by adding custom SQL which annotates each record with
        the score from the search 'hit'. This is brittle, caveat emptor.

        The RawSQL clause is in the form:

            SELECT CASE {{model}}.id WHEN {{id}} THEN {{score}} END

        The "WHEN x THEN y" is repeated for every hit. The resulting SQL, in
        full is like this:

            SELECT "freelancer_freelancerprofile"."id",
                (SELECT CASE freelancer_freelancerprofile.id
                    WHEN 25 THEN 1.0
                    WHEN 26 THEN 1.0
                    [...]
                    ELSE 0
                END) AS "search_score"
            FROM "freelancer_freelancerprofile"
            WHERE "freelancer_freelancerprofile"."id" IN (25, 26, [...])
            ORDER BY "search_score" DESC

        It should be very fast, as there is no table lookup, but there is an
        assumption at the heart of this, which is that the search query doesn't
        contain the entire database - i.e. that it has been paged. (ES itself
        caps the results at 10,000.)

        """
        hits = search_query.hits
        score_sql = self._raw_sql([(h["id"], h["score"] or 0) for h in hits])
        rank_sql = self._raw_sql([(hits[i]["id"], i) for i in range(len(hits))])
        return (
            self.get_queryset()
            .filter(pk__in=[h["id"] for h in hits])
            # add the query relevance score
            .annotate(search_score=RawSQL(score_sql, ()))
            # add the ordering number (0-based)
            .annotate(search_rank=RawSQL(rank_sql, ()))
            .order_by("search_rank")
        )