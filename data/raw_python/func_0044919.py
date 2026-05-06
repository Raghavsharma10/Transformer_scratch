def part_show(self, part, printout=False):
        '''
        Opens/shows the aggregator's URI for the part.


        printout: if set, only printout the URI, do not open the browser.
        '''

        result = self._e.parts_match(
            queries=[{'mpn_or_sku': part}],
            exact_only=True,
            show_mpn=True,
            show_octopart_url=True
        )
        if result[1][0]['hits'] == 0:
            print("No result")
            return ReturnValues.NO_RESULTS
        result = result[1][0]['items'][0]
        if not printout:
            print("Opening page for part '{}'.".format(result['mpn']))
            webbrowser.open(result['octopart_url'], 2)
        else:
            print("Webpage for part '{}':".format(result['mpn']))
            print("    → URL:      {}".format(result['octopart_url']))
        return ReturnValues.OK