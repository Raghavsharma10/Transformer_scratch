def _initializeLocationCache(self):
        """
        CGD uses Faldo ontology for locations, it's a bit complicated.
        This function sets up an in memory cache of all locations, which
        can be queried via:
        locationMap[build][chromosome][begin][end] = location["_id"]
        """
        # cache of locations
        self._locationMap = {}
        locationMap = self._locationMap
        triples = self._rdfGraph.triples
        Ref = rdflib.URIRef

        associations = []
        for subj, _, _ in triples((None, RDF.type, Ref(ASSOCIATION))):
            associations.append(subj.toPython())

        locationIds = []
        for association in associations:
            for _, _, obj in triples((Ref(association),
                                      Ref(HAS_SUBJECT), None)):
                locationIds.append(obj.toPython())

        locations = []
        for _id in locationIds:
            location = {}
            location["_id"] = _id
            for subj, predicate, obj in triples((Ref(location["_id"]),
                                                 None, None)):
                if not predicate.toPython() in location:
                    location[predicate.toPython()] = []
                bisect.insort(location[predicate.toPython()], obj.toPython())
                if FALDO_LOCATION in location:
                    locations.append(location)

        for location in locations:
            for _id in location[FALDO_LOCATION]:
                # lookup faldo region, ensure positions are sorted
                faldoLocation = {}
                faldoLocation["_id"] = _id
                for subj, predicate, obj in triples((Ref(faldoLocation["_id"]),
                                                    None, None)):
                    if not predicate.toPython() in faldoLocation:
                        faldoLocation[predicate.toPython()] = []
                    bisect.insort(faldoLocation[predicate.toPython()],
                                  obj.toPython())

                faldoBegins = []

                for _id in faldoLocation[FALDO_BEGIN]:
                    faldoBegin = {}
                    faldoBegin["_id"] = _id
                    for subj, predicate, obj in triples(
                                                (Ref(faldoBegin["_id"]),
                                                    None, None)):
                        faldoBegin[predicate.toPython()] = obj.toPython()
                    faldoBegins.append(faldoBegin)

                faldoReferences = []
                for _id in faldoLocation[FALDO_BEGIN]:
                    faldoReference = {}
                    faldoReference["_id"] = faldoBegin[FALDO_REFERENCE]
                    for subj, predicate, obj in triples(
                                                (Ref(faldoReference["_id"]),
                                                    None, None)):
                        faldoReference[predicate.toPython()] = obj.toPython()
                    faldoReferences.append(faldoReference)

                faldoEnds = []
                for _id in faldoLocation[FALDO_END]:
                    faldoEnd = {}
                    faldoEnd["_id"] = _id
                    for subj, predicate, obj in triples((Ref(faldoEnd["_id"]),
                                                        None, None)):
                        faldoEnd[predicate.toPython()] = obj.toPython()
                    faldoEnds.append(faldoEnd)

                for idx, faldoReference in enumerate(faldoReferences):
                    if MEMBER_OF in faldoReference:
                        build = faldoReference[MEMBER_OF].split('/')[-1]
                        chromosome = faldoReference[LABEL].split(' ')[0]
                        begin = faldoBegins[idx][FALDO_POSITION]
                        end = faldoEnds[idx][FALDO_POSITION]
                        if build not in locationMap:
                            locationMap[build] = {}
                        if chromosome not in locationMap[build]:
                            locationMap[build][chromosome] = {}
                        if begin not in locationMap[build][chromosome]:
                            locationMap[build][chromosome][begin] = {}
                        if end not in locationMap[build][chromosome][begin]:
                            locationMap[build][chromosome][begin][end] = {}
                        locationMap[build][chromosome][begin][end] = \
                            location["_id"]
                        locationMap[location["_id"]] = {
                            "build": build,
                            "chromosome": chromosome,
                            "begin": begin,
                            "end": end,
                        }