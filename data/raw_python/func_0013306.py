def _getAnnotationAnalysis(self, varFile):
        """
        Assembles metadata within the VCF header into a GA4GH Analysis object.

        :return: protocol.Analysis
        """
        header = varFile.header
        analysis = protocol.Analysis()
        formats = header.formats.items()
        infos = header.info.items()
        filters = header.filters.items()
        for prefix, content in [("FORMAT", formats), ("INFO", infos),
                                ("FILTER", filters)]:
            for contentKey, value in content:
                key = "{0}.{1}".format(prefix, value.name)
                if key not in analysis.attributes.attr:
                    analysis.attributes.attr[key].Clear()
                if value.description is not None:
                    analysis.attributes.attr[
                        key].values.add().string_value = value.description
        analysis.created = self._creationTime
        analysis.updated = self._updatedTime
        for r in header.records:
            # Don't add a key to info if there's nothing in the value
            if r.value is not None:
                if r.key not in analysis.attributes.attr:
                    analysis.attributes.attr[r.key].Clear()
                analysis.attributes.attr[r.key] \
                    .values.add().string_value = str(r.value)
            if r.key == "created" or r.key == "fileDate":
                # TODO handle more date formats
                try:
                    if '-' in r.value:
                        fmtStr = "%Y-%m-%d"
                    else:
                        fmtStr = "%Y%m%d"
                    analysis.created = datetime.datetime.strptime(
                        r.value, fmtStr).isoformat() + "Z"
                except ValueError:
                    # is there a logger we should tell?
                    # print("INFO: Could not parse variant annotation time")
                    pass  # analysis.create_date_time remains datetime.now()
            if r.key == "software":
                analysis.software.append(r.value)
            if r.key == "name":
                analysis.name = r.value
            if r.key == "description":
                analysis.description = r.value
        analysis.id = str(datamodel.VariantAnnotationSetAnalysisCompoundId(
            self._compoundId, "analysis"))
        return analysis