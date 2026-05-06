def part_datasheet(self, part, command=None, path=None):
        '''
        downloads and/or shows the datasheet of a given part

        command: if set will use it to open the datasheet.
        path: if set will download the file under that path.

        if path is given alone, the file will only get downloaded,
        if command is given alone, the file will be downloaded in a temporary
        folder, which will be destroyed just after being opened.
        if both path and command are given, the file will be downloaded and
        stored in the chosen location.
        '''
        result = self._e.parts_match(
            queries=[{'mpn_or_sku': part}],
            exact_only=True,
            show_mpn=True,
            show_datasheets=True,
            include_datasheets=True
        )
        if result[1][0]['hits'] == 0:
            print("No result")
            return ReturnValues.NO_RESULTS

        result = result[1][0]['items'][0]
        print("Downloading datasheet for '{}':".format(result['mpn']))
        try:
            if len(result['datasheets']) > 0:
                for datasheet in result['datasheets']:
                    if not path:
                        path = tempfile.mkdtemp()
                    out = path+'/'+result['mpn']+'-'+datasheet['url'].split('/')[-1]
                    download_file(datasheet['url'], out)
                    print('Datasheet file saved as {}.'.format(out))
                    if command:
                        subprocess.call([command, out])
        finally:
            if not path:
                shutil.rmtree(path)
        return ReturnValues.OK