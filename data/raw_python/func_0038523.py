def get_reading_list_context(self, **kwargs):
        """Returns the context dictionary for a given reading list."""
        reading_list = None
        context = {
            "name": "",
            "content": reading_list,
            "targeting": {},
            "videos": []
        }

        if self.reading_list_identifier == "popular":
            reading_list = popular_content()
            context.update({"name": self.reading_list_identifier})

            # Popular is augmented.
            reading_list = self.augment_reading_list(reading_list)
            context.update({"content": reading_list})
            return context

        if self.reading_list_identifier.startswith("specialcoverage"):
            special_coverage = SpecialCoverage.objects.get_by_identifier(
                self.reading_list_identifier
            )
            reading_list = special_coverage.get_content().query(
                SponsoredBoost(field_name="tunic_campaign_id")
            ).sort("_score", "-published")
            context["targeting"]["dfp_specialcoverage"] = special_coverage.slug
            if special_coverage.tunic_campaign_id:
                context["tunic_campaign_id"] = special_coverage.tunic_campaign_id
                context["targeting"].update({
                    "dfp_campaign_id": special_coverage.tunic_campaign_id
                })
                # We do not augment sponsored special coverage lists.
                reading_list = self.update_reading_list(reading_list)
            else:
                reading_list = self.augment_reading_list(reading_list)
            context.update({
                "name": special_coverage.name,
                "videos": special_coverage.videos,
                "content": reading_list
            })
            return context

        if self.reading_list_identifier.startswith("section"):
            section = Section.objects.get_by_identifier(self.reading_list_identifier)
            reading_list = section.get_content()
            reading_list = self.augment_reading_list(reading_list)
            context.update({
                "name": section.name,
                "content": reading_list
            })
            return context

        reading_list = Content.search_objects.search()
        reading_list = self.augment_reading_list(reading_list, reverse_negate=True)
        context.update({
            "name": "Recent News",
            "content": reading_list
        })
        return context