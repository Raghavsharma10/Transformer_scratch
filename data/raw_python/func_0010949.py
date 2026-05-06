def kill_eaters(self):
        """
        Returns a list of tuples containing the proper localized kill eater type strings and their values
        according to set/type/value "order"
        """

        eaters = {}
        ranktypes = self._kill_types

        for attr in self:
            aname = attr.name.strip()
            aid = attr.id

            if aname.startswith("kill eater"):
                try:
                    # Get the name prefix (matches up type and score and
                    # determines the primary type for ranking)
                    eateri = list(filter(None, aname.split(' ')))[-1]
                    if eateri.isdigit():
                        eateri = int(eateri)
                    else:
                        # Probably the primary type/score which has no number
                        eateri = 0
                except IndexError:
                    # Fallback to attr ID (will completely fail to make
                    # anything legible but better than nothing)
                    eateri = aid

                if aname.find("user") != -1:
                    # User score types have lower sorting priority
                    eateri += 100

                eaters.setdefault(eateri, [None, None])
                if aname.find("score type") != -1 or aname.find("kill type") != -1:
                    # Score type attribute
                    if eaters[eateri][0] is None:
                        eaters[eateri][0] = attr.value
                else:
                    # Value attribute
                    eaters[eateri][1] = attr.value

        eaterlist = []
        defaultleveldata = "KillEaterRank"
        for key, eater in sorted(eaters.items()):
            etype, count = eater

            # Eater type can be null (it still is in some older items), null
            # count means we're looking at either an uninitialized item or
            # schema item
            if count is not None:
                rank = ranktypes.get(etype or 0,
                                     {"level_data": defaultleveldata,
                                      "type_name": "Count"})
                eaterlist.append((rank.get("level_data", defaultleveldata),
                                  rank["type_name"], count))

        return eaterlist