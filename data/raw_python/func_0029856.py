def record_to_objects(self):
        """Create config records to match the file metadata"""
        from ambry.orm.exc import NotFoundError

        fr = self.record

        contents = fr.unpacked_contents

        if not contents:
            return

        # Zip transposes an array when in the form of a list of lists, so this transposes so
        # each row starts with the heading and the rest of the row are the values
        # for that row. The bool and filter return false when none of the values
        # are non-empty. Then zip again to transpose to original form.

        non_empty_rows = drop_empty(contents)

        s = self._dataset._database.session

        for i, row in enumerate(non_empty_rows):

            if i == 0:
                header = row
            else:
                d = dict(six.moves.zip(header, row))

                if 'widths' in d:
                    del d['widths']  # Obsolete column in old spreadsheets.

                if 'table' in d:
                    d['dest_table_name'] = d['table']
                    del d['table']

                if 'order' in d:
                    d['stage'] = d['order']
                    del d['order']

                if 'dest_table' in d:
                    d['dest_table_name'] = d['dest_table']
                    del d['dest_table']

                if 'source_table' in d:
                    d['source_table_name'] = d['source_table']
                    del d['source_table']

                d['d_vid'] = self._dataset.vid

                d['state'] = 'synced'

                try:
                    ds = self._dataset.source_file(str(d['name']))
                    ds.update(**d)
                except NotFoundError:
                    name = d['name']
                    del d['name']

                    try:
                        ds = self._dataset.new_source(name, **d)
                    except:
                        print(name, d)
                        import pprint
                        pprint.pprint(d)
                        raise
                except:  # Odd error with 'none' in keys for d
                    print('!!!', header)
                    print('!!!', row)
                    raise

                s.merge(ds)

        self._dataset._database.commit()