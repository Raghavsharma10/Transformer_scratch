def read_binary(self, ba, param_groups=None):
        """
        ba - binaryDataArray XML node
        """
        if ba is None:
            return []

        pgr = ba.find('m:referenceableParamGroupRef', namespaces=self.ns)
        if pgr is not None and param_groups is not None:
            q = 'm:referenceableParamGroup[@id="' + pgr.get('ref') + '"]'
            pg = param_groups.find(q, namespaces=self.ns)
        else:
            pg = ba

        if pg.find('m:cvParam[@accession="MS:1000574"]',
                   namespaces=self.ns) is not None:
            compress = True
        elif pg.find('m:cvParam[@accession="MS:1000576"]',
                     namespaces=self.ns) is not None:
            compress = False
        else:
            # TODO: no info? should check the other record?
            pass

        if pg.find('m:cvParam[@accession="MS:1000521"]',
                   namespaces=self.ns) is not None:
            dtype = 'f'
        elif pg.find('m:cvParam[@accession="MS:1000523"]',
                     namespaces=self.ns) is not None:
            dtype = 'd'
        else:
            # TODO: no info? should check the other record?
            pass

        datatext = ba.find('m:binary', namespaces=self.ns).text
        if compress:
            rawdata = zlib.decompress(base64.b64decode(datatext))
        else:
            rawdata = base64.b64decode(datatext)
        return np.fromstring(rawdata, dtype=dtype)