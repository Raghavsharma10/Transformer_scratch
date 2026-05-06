def export_txt(cls, feed):
        '''Export records as a GTFS comma-separated file'''
        objects = cls.objects.in_feed(feed)

        # If no records, return None
        if not objects.exists():
            return

        # Get the columns used in the dataset
        column_map = objects.populated_column_map()
        columns, fields = zip(*column_map)
        extra_columns = feed.meta.get(
            'extra_columns', {}).get(cls.__name__, [])

        # Get sort order
        if hasattr(cls, '_sort_order'):
            sort_fields = cls._sort_order
        else:
            sort_fields = []
            for field in fields:
                base_field = field.split('__', 1)[0]
                point_match = re_point.match(base_field)
                if point_match:
                    continue
                field_type = cls._meta.get_field(base_field)
                assert not isinstance(field_type, ManyToManyField)
                sort_fields.append(field)

        # Create CSV writer
        out = StringIO()
        csv_writer = writer(out, lineterminator='\n')

        # Write header row
        header_row = [text_type(c) for c in columns]
        header_row.extend(extra_columns)
        write_text_rows(csv_writer, [header_row])

        # Report the work to be done
        total = objects.count()
        logger.info(
            '%d %s to export...',
            total, cls._meta.verbose_name_plural)

        # Populate related items cache
        model_to_field_name = {}
        cache = {}
        for field_name in fields:
            if '__' in field_name:
                local_field_name, subfield_name = field_name.split('__', 1)
                field = cls._meta.get_field(local_field_name)
                field_type = field.related_model
                model_name = field_type.__name__
                if model_name in model_to_field_name:
                    # Already loaded this model under a different field name
                    cache[field_name] = cache[model_to_field_name[model_name]]
                else:
                    # Load all feed data for this model
                    pairs = field_type.objects.in_feed(
                        feed).values_list('id', subfield_name)
                    cache[field_name] = dict(
                        (i, text_type(x)) for i, x in pairs)
                    cache[field_name][None] = u''
                    model_to_field_name[model_name] = field_name

        # Assemble the rows, writing when we hit batch size
        count = 0
        rows = []
        for item in objects.order_by(*sort_fields).iterator():
            row = []
            for csv_name, field_name in column_map:
                obj = item
                point_match = re_point.match(field_name)
                if '__' in field_name:
                    # Return relations from cache
                    local_field_name = field_name.split('__', 1)[0]
                    field_id = getattr(obj, local_field_name + '_id')
                    row.append(cache[field_name][field_id])
                elif point_match:
                    # Get the lat or long from the point
                    name, index = point_match.groups()
                    field = getattr(obj, name)
                    row.append(field.coords[int(index)])
                else:
                    # Handle other field types
                    field = getattr(obj, field_name) if obj else ''
                    if isinstance(field, date):
                        formatted = field.strftime(u'%Y%m%d')
                        row.append(text_type(formatted))
                    elif isinstance(field, bool):
                        row.append(1 if field else 0)
                    elif field is None:
                        row.append(u'')
                    else:
                        row.append(text_type(field))
            for col in extra_columns:
                row.append(obj.extra_data.get(col, u''))
            rows.append(row)
            if len(rows) % batch_size == 0:  # pragma: no cover
                write_text_rows(csv_writer, rows)
                count += len(rows)
                logger.info(
                    "Exported %d %s",
                    count, cls._meta.verbose_name_plural)
                rows = []

        # Write rows smaller than batch size
        write_text_rows(csv_writer, rows)
        return out.getvalue()