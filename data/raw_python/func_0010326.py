def to_xml(self):
        """Convert this datapoint into a form suitable for pushing to device cloud

        An XML string will be returned that will contain all pieces of information
        set on this datapoint.  Values not set (e.g. quality) will be ommitted.

        """
        type_converter = _get_encoder_method(self._data_type)
        # Convert from python native to device cloud
        encoded_data = type_converter(self._data)

        out = StringIO()
        out.write("<DataPoint>")
        out.write("<streamId>{}</streamId>".format(self.get_stream_id()))
        out.write("<data>{}</data>".format(encoded_data))
        conditional_write(out, "<description>{}</description>", self.get_description())
        if self.get_timestamp() is not None:
            out.write("<timestamp>{}</timestamp>".format(isoformat(self.get_timestamp())))
        conditional_write(out, "<quality>{}</quality>", self.get_quality())
        if self.get_location() is not None:
            out.write("<location>%s</location>" % ",".join(map(str, self.get_location())))
        conditional_write(out, "<streamType>{}</streamType>", self.get_data_type())
        conditional_write(out, "<streamUnits>{}</streamUnits>", self.get_units())
        out.write("</DataPoint>")
        return out.getvalue()