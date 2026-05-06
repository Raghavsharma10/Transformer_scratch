def read(filehandle, source):
        """Read data into a :class:`~nmrstarlib.nmrstarlib.StarFile` instance.

        :param filehandle: file-like object.
        :type filehandle: :py:class:`io.TextIOWrapper`, :py:class:`gzip.GzipFile`,
                          :py:class:`bz2.BZ2File`, :py:class:`zipfile.ZipFile`
        :param str source: String indicating where file is coming from (path, url).
        :return: subclass of :class:`~nmrstarlib.nmrstarlib.StarFile`.
        :rtype: :class:`~nmrstarlib.nmrstarlib.NMRStarFile` or :class:`~nmrstarlib.nmrstarlib.CIFFile`
        """
        input_str = filehandle.read()
        nmrstar_str = StarFile._is_nmrstar(input_str)
        cif_str = StarFile._is_cif(input_str)
        json_str = StarFile._is_json(input_str)

        if not input_str:
            pass

        elif nmrstar_str:
            starfile = NMRStarFile(source)
            starfile._build_file(nmrstar_str)
            filehandle.close()
            return starfile

        elif cif_str:
            starfile = CIFFile(source)
            starfile._build_file(cif_str)
            filehandle.close()
            return starfile

        elif json_str:
            if u"save_" in json_str:
                starfile = NMRStarFile(source)
                starfile.update(json.loads(json_str, object_pairs_hook=OrderedDict))
                starfile.id = starfile[u"data"]
                filehandle.close()
                return starfile

            elif u"entry.id" in json_str:
                starfile = CIFFile(source)
                starfile.update(json.loads(json_str, object_pairs_hook=OrderedDict))
                starfile.id = starfile[u"data"]
                filehandle.close()
                return starfile
            else:
                raise TypeError("Unknown file format")
        else:
            raise TypeError("Unknown file format")