def _insert_or_update(self, resourcetype, source, mode='insert', hhclass='Service'):
        """
        Insert or update a record in the repository
        """

        keywords = []

        if self.filter is not None:
            catalog = Catalog.objects.get(id=int(self.filter.split()[-1]))

        try:
            if hhclass == 'Layer':
                # TODO: better way of figuring out duplicates
                match = Layer.objects.filter(name=source.name,
                                             title=source.title,
                                             abstract=source.abstract,
                                             is_monitored=False)
                matches = match.all()
                if matches:
                    if mode == 'insert':
                        raise RuntimeError('HHypermap error: Layer %d \'%s\' already exists' % (
                                           matches[0].id, source.title))
                    elif mode == 'update':
                        match.update(
                            name=source.name,
                            title=source.title,
                            abstract=source.abstract,
                            is_monitored=False,
                            xml=source.xml,
                            wkt_geometry=source.wkt_geometry,
                            anytext=util.get_anytext([source.title, source.abstract, source.keywords_csv])
                        )

                service = get_service(source.xml)
                res, keywords = create_layer_from_metadata_xml(resourcetype, source.xml,
                                                               monitor=False, service=service,
                                                               catalog=catalog)

                res.save()

                LOGGER.debug('Indexing layer with id %s on search engine' % res.uuid)
                index_layer(res.id, use_cache=True)

            else:
                if resourcetype == 'http://www.opengis.net/cat/csw/2.0.2':
                    res = Endpoint(url=source, catalog=catalog)
                else:
                    res = Service(type=HYPERMAP_SERVICE_TYPES[resourcetype], url=source, catalog=catalog)

                res.save()

            if keywords:
                for kw in keywords:
                    res.keywords.add(kw)
        except Exception as err:
            raise RuntimeError('HHypermap error: %s' % err)

        # return a list of ids that were inserted or updated
        ids = []

        if hhclass == 'Layer':
            ids.append({'identifier': res.uuid, 'title': res.title})
        else:
            if resourcetype == 'http://www.opengis.net/cat/csw/2.0.2':
                for res in Endpoint.objects.filter(url=source).all():
                    ids.append({'identifier': res.uuid, 'title': res.url})
            else:
                for res in Service.objects.filter(url=source).all():
                    ids.append({'identifier': res.uuid, 'title': res.title})

        return ids