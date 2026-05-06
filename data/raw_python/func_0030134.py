def set_coverage(self, stats):
        """"Extract time space and grain coverage from the stats and store them in the partition"""
        from ambry.util.datestimes import expand_to_years

        scov = set()
        tcov = set()
        grains = set()

        def summarize_maybe(gvid):
            try:
                return parse_to_gvid(gvid).summarize()
            except:
                return None

        def simplifiy_maybe(values, column):

            parsed = []

            for gvid in values:
                # The gvid should not be a st
                if gvid is None or gvid == 'None':
                    continue
                try:
                    parsed.append(parse_to_gvid(gvid))
                except ValueError as e:
                    if self._bundle:
                        self._bundle.warn("While analyzing geo coverage in final partition stage, " +
                                           "Failed to parse gvid '{}' in {}.{}: {}"
                                           .format(str(gvid), column.table.name, column.name, e))

            try:
                return isimplify(parsed)
            except:
                return None

        def int_maybe(year):
            try:
                return int(year)
            except:
                return None

        for c in self.table.columns:

            if c.name not in stats:
                continue

            try:
                if stats[c.name].is_gvid or stats[c.name].is_geoid:
                    scov |= set(x for x in simplifiy_maybe(stats[c.name].uniques, c))
                    grains |= set(summarize_maybe(gvid) for gvid in stats[c.name].uniques)

                elif stats[c.name].is_year:
                    tcov |= set(int_maybe(x) for x in stats[c.name].uniques)

                elif stats[c.name].is_date:
                    # The fuzzy=True argument allows ignoring the '-' char in dates produced by .isoformat()
                    try:
                        tcov |= set(parser.parse(x, fuzzy=True).year if isinstance(x, string_types) else x.year for x in
                                    stats[c.name].uniques)
                    except ValueError:
                        pass

            except Exception as e:
                self._bundle.error("Failed to set coverage for column '{}', partition '{}': {}"
                                   .format(c.name, self.identity.vname, e))
                raise

        # Space Coverage

        if 'source_data' in self.data:

            for source_name, source in list(self.data['source_data'].items()):
                scov.add(self.parse_gvid_or_place(source['space']))

        if self.identity.space:  # And from the partition name
            try:
                scov.add(self.parse_gvid_or_place(self.identity.space))
            except ValueError:
                # Couldn't parse the space as a GVid
                pass

        # For geo_coverage, only includes the higher level summary levels, counties, states,
        # places and urban areas.
        self.space_coverage = sorted([str(x) for x in scov if bool(x) and x.sl
                                      in (10, 40, 50, 60, 160, 400)])

        #
        # Time Coverage

        # From the source
        # If there was a time value in the source that this partition was created from, then
        # add it to the years.
        if 'source_data' in self.data:
            for source_name, source in list(self.data['source_data'].items()):
                if 'time' in source:
                    for year in expand_to_years(source['time']):
                        if year:
                            tcov.add(year)

        # From the partition name
        if self.identity.name.time:
            for year in expand_to_years(self.identity.name.time):
                if year:
                    tcov.add(year)

        self.time_coverage = [t for t in tcov if t]

        #
        # Grains

        if 'source_data' in self.data:
            for source_name, source in list(self.data['source_data'].items()):
                if 'grain' in source:
                    grains.add(source['grain'])

        self.grain_coverage = sorted(str(g) for g in grains if g)