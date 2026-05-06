def check_request_parameters(self, parameters: dict = dict):
        """Check parameters passed to avoid errors and help debug.

        :param dict response: search request parameters
        """
        # -- SEMANTIC QUERY ---------------------------------------------------
        li_args = parameters.get("q").split()
        logging.debug(li_args)

        # Unicity
        li_filters = [i.split(":")[0] for i in li_args]
        filters_count = Counter(li_filters)
        li_filters_must_be_unique = ("coordinate-system", "format", "owner", "type")
        for i in filters_count:
            if i in li_filters_must_be_unique and filters_count.get(i) > 1:
                raise ValueError(
                    "This query filter must be unique: {}"
                    " and it occured {} times.".format(i, filters_count.get(i))
                )

        # dict
        dico_query = FILTER_KEYS.copy()
        for i in li_args:
            if i.startswith("action"):
                dico_query["action"].append(i.split(":")[1:][0])
                continue
            elif i.startswith("catalog"):
                dico_query["catalog"].append(i.split(":")[1:][0])
                continue
            elif i.startswith("contact") and i.split(":")[1] == "group":
                dico_query["contact:group"].append(i.split(":")[1:][1])
                continue
            elif i.startswith("contact"):
                dico_query["contact:isogeo"].append(i.split(":", 1)[1])
                continue
            elif i.startswith("coordinate-system"):
                dico_query["coordinate-system"].append(i.split(":")[1:][0])
                continue
            elif i.startswith("data-source"):
                dico_query["data-source"].append(i.split(":")[1:][0])
                continue
            elif i.startswith("format"):
                dico_query["format"].append(i.split(":")[1:][0])
                continue
            elif i.startswith("has-no"):
                dico_query["has-no"].append(i.split(":")[1:][0])
                continue
            elif i.startswith("keyword:isogeo"):
                dico_query["keyword:isogeo"].append(i.split(":")[1:][1])
                continue
            elif i.startswith("keyword:inspire-theme"):
                dico_query["keyword:inspire-theme"].append(i.split(":")[1:][1])
                continue
            elif i.startswith("license:isogeo"):
                dico_query["license:isogeo"].append(i.split(":")[1:][1:])
                continue
            elif i.startswith("license"):
                dico_query["license:group"].append(i.split(":", 1)[1:][0:])
                continue
            elif i.startswith("owner"):
                dico_query["owner"].append(i.split(":")[1:][0])
                continue
            elif i.startswith("provider"):
                dico_query["provider"].append(i.split(":")[1:][0])
                continue
            elif i.startswith("share"):
                dico_query["share"].append(i.split(":")[1:][0])
                continue
            elif i.startswith("type"):
                dico_query["type"].append(i.split(":")[1:][0])
                continue
            else:
                # logging.debug(i.split(":")[1], i.split(":")[1].isdigit())
                dico_query["text"].append(i)
                continue

        # Values
        dico_filters = {i.split(":")[0]: i.split(":")[1:] for i in li_args}
        if dico_filters.get("type", ("dataset",))[0].lower() not in FILTER_TYPES:
            raise ValueError(
                "type value must be one of: {}".format(" | ".join(FILTER_TYPES))
            )
        elif dico_filters.get("action", ("download",))[0].lower() not in FILTER_ACTIONS:
            raise ValueError(
                "action value must be one of: {}".format(" | ".join(FILTER_ACTIONS))
            )
        elif (
            dico_filters.get("provider", ("manual",))[0].lower() not in FILTER_PROVIDERS
        ):
            raise ValueError(
                "provider value must be one of: {}".format(" | ".join(FILTER_PROVIDERS))
            )
        else:
            logging.debug(dico_filters)

        # -- GEOGRAPHIC -------------------------------------------------------
        in_box = parameters.get("box")
        in_geo = parameters.get("geo")
        # geometric relation
        in_rel = parameters.get("rel")
        if in_rel and in_box is None and in_geo is None:
            raise ValueError("'rel' should'nt be used without box or geo.")
        elif in_rel not in GEORELATIONS and in_rel is not None:
            raise ValueError(
                "{} is not a correct value for 'georel'."
                " Must be one of: {}.".format(in_rel, " | ".join(GEORELATIONS))
            )