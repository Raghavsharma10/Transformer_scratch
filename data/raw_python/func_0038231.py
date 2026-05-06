def percolate_special_coverage(self, max_size=10, sponsored_only=False):
        """gets list of active, sponsored special coverages containing this content via
        Elasticsearch Percolator (see SpecialCoverage._save_percolator)

        Sorting:
            1) Manually added
            2) Most recent start date
        """

        # Elasticsearch v1.4 percolator range query does not support DateTime range queries
        # (PercolateContext.nowInMillisImpl is not implemented). Once using
        # v1.6+ we can instead compare "start_date/end_date" to python DateTime
        now_epoch = datetime_to_epoch_seconds(timezone.now())

        MANUALLY_ADDED_BOOST = 10
        SPONSORED_BOOST = 100  # Must be order of magnitude higher than "Manual" boost

        # Unsponsored boosting to either lower priority or exclude
        if sponsored_only:
            # Omit unsponsored
            unsponsored_boost = 0
        else:
            # Below sponsored (inverse boost, since we're filtering on "sponsored=False"
            unsponsored_boost = (1.0 / SPONSORED_BOOST)

        # ES v1.4 has more limited percolator capabilities than later
        # implementations. As such, in order to get this to work, we need to
        # sort via scoring_functions, and then manually filter out zero scores.
        sponsored_filter = {
            "query": {
                "function_score": {
                    "functions": [

                        # Boost Recent Special Coverage
                        # Base score is start time
                        # Note: ES 1.4 sorting granularity is poor for times
                        # within 1 hour of each other.
                        {

                            # v1.4 "field_value_factor" does not yet support
                            # "missing" param, and so must filter on whether
                            # "start_date" field exists.
                            "filter": {
                                "exists": {
                                    "field": "start_date",
                                },
                            },
                            "field_value_factor": {
                                "field": "start_date",
                            }
                        },
                        {
                            # Related to above, if "start_date" not found, omit
                            # via zero score.
                            "filter": {
                                "not": {
                                    "exists": {
                                        "field": "start_date",
                                    },
                                },
                            },
                            "weight": 0,
                        },


                        # Ignore non-special-coverage percolator entries
                        {
                            "filter": {
                                "not": {
                                    "prefix": {"_id": "specialcoverage"},
                                },
                            },
                            "weight": 0,
                        },

                        # Boost Manually Added Content
                        {
                            "filter": {
                                "terms": {
                                    "included_ids": [self.id],
                                }
                            },
                            "weight": MANUALLY_ADDED_BOOST,
                        },
                        # Penalize Inactive (Zero Score Will be Omitted)
                        {
                            "filter": {
                                "or": [
                                    {
                                        "range": {
                                            "start_date_epoch": {
                                                "gt": now_epoch,
                                            },
                                        }
                                    },
                                    {
                                        "range": {
                                            "end_date_epoch": {
                                                "lte": now_epoch,
                                            },
                                        }
                                    },
                                ],
                            },
                            "weight": 0,
                        },
                        # Penalize Unsponsored (will either exclude or lower
                        # based on "sponsored_only" flag)
                        {
                            "filter": {
                                "term": {
                                    "sponsored": False,
                                }
                            },
                            "weight": unsponsored_boost,
                        },
                    ],
                },
            },

            "sort": "_score",  # The only sort method supported by ES v1.4 percolator
            "size": max_size,  # Required for sort
        }

        results = _percolate(index=self.mapping.index,
                             doc_type=self.mapping.doc_type,
                             content_id=self.id,
                             body=sponsored_filter)

        return [r["_id"] for r in results
                # Zero score used to omit results via scoring function (ex: inactive)
                if r['_score'] > 0]