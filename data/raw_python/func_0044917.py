def part_specs(self, part):
        '''
        returns the specifications of the given part. If multiple parts are
        matched, only the first one will be output.

        part: the productname or sku

        prints the results on stdout
        '''
        result = self._e.parts_match(
            queries=[{'mpn_or_sku': part}],
            exact_only=True,
            show_mpn=True,
            show_manufacturer=True,
            show_octopart_url=True,
            show_short_description=True,
            show_specs=True,
            show_category_uids=True,
            show_external_links=True,
            show_reference_designs=True,
            show_cad_models=True,
            show_datasheets=True,
            include_specs=True,
            include_category_uids=True,
            include_external_links=True,
            include_reference_designs=True,
            include_cad_models=True,
            include_datasheets=True
        )
        if result[1][0]['hits'] == 0:
            print("No result")
            return ReturnValues.NO_RESULTS

        result = result[1][0]['items'][0]
        print("Showing specs for '{}':".format(result['mpn']))
        print(" → Manufacturer:      {}".format(result['manufacturer']['name']))
        print("  → Specifications:    ")
        for k,v in result['specs'].items():
            name = v['metadata']['name'] if v['metadata']['name'] else k
            min_value = v['min_value'] if v['min_value'] else ''
            max_value = v['max_value'] if v['max_value'] else ''
            unit = ' ({})'.format(v['metadata']['unit']['name']) if v['metadata']['unit'] else ''
            value = ','.join(v['value']) if len(v['value']) > 0 else ''

            if value and not (min_value or max_value):
                print("    → {:20}: {}{}".format(name, value, unit))
            elif value and min_value and max_value:
                print("    → {:20}: {}{} (min: {}, max: {})".format(name, value, unit, min_value, max_value))
            elif not value and min_value and max_value:
                print("    → {:20}:{} min: {}, max: {}".format(name, unit, min_value, max_value))
            elif not value and min_value and not max_value:
                print("    → {:20}:{} min: {}".format(name, unit, min_value))
            elif not value and not min_value and max_value:
                print("    → {:20}:{} max: {}".format(name, unit, max_value))

        print(" → URI:               {}".format(result['octopart_url']))
        if result['external_links']['evalkit_url'] \
                or result['external_links']['freesample_url'] \
                or result['external_links']['product_url']:
            print("  → External Links")
            if result['external_links']['evalkit_url']:
                print("    → Evaluation kit: {}".format(result['external_links']['evalkit_url']))
            if result['external_links']['freesample_url']:
                print("    → Free Sample: {}".format(result['external_links']['freesample_url']))
            if result['external_links']['product_url']:
                print("    → Product URI: {}".format(result['external_links']['product_url']))
        if len(result['datasheets']) > 0:
            print("  → Datasheets")
            for datasheet in result['datasheets']:
                print("    → URL:      {}".format(datasheet['url']))
                if datasheet['metadata']:
                    print("      → Updated:  {}".format(datasheet['metadata']['last_updated']))
                    print("      → Nb Pages: {}".format(datasheet['metadata']['num_pages']))
        if len(result['reference_designs']) > 0:
            print("  → Reference designs: ")
        if len(result['cad_models']) > 0:
            print("  → CAD Models:        ")
        return ReturnValues.OK