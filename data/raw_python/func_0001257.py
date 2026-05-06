def pg2ogr(
        self,
        sql,
        driver,
        outfile,
        outlayer=None,
        column_remap=None,
        s_srs="EPSG:3005",
        t_srs=None,
        geom_type=None,
        append=False,
    ):
        """
        A wrapper around ogr2ogr, for quickly dumping a postgis query to file.
        Suppported formats are ["ESRI Shapefile", "GeoJSON", "FileGDB", "GPKG"]
           - for GeoJSON, transforms to EPSG:4326
           - for Shapefile, consider supplying a column_remap dict
           - for FileGDB, geom_type is required
             (https://trac.osgeo.org/gdal/ticket/4186)
        """
        if driver == "FileGDB" and geom_type is None:
            raise ValueError("Specify geom_type when writing to FileGDB")
        filename, ext = os.path.splitext(os.path.basename(outfile))
        if not outlayer:
            outlayer = filename
        u = urlparse(self.url)
        pgcred = "host={h} user={u} dbname={db} password={p}".format(
            h=u.hostname, u=u.username, db=u.path[1:], p=u.password
        )
        # use a VRT so we can remap columns if a lookoup is provided
        if column_remap:
            # if specifiying output field names, all fields have to be specified
            # rather than try and parse the input sql, just do a test run of the
            # query and grab column names from that
            columns = [c for c in self.query(sql).keys() if c != "geom"]
            # make sure all columns are represented in the remap
            for c in columns:
                if c not in column_remap.keys():
                    column_remap[c] = c
            field_remap_xml = " \n".join(
                [
                    '<Field name="' + column_remap[c] + '" src="' + c + '"/>'
                    for c in columns
                ]
            )
        else:
            field_remap_xml = ""
        vrt = """<OGRVRTDataSource>
                   <OGRVRTLayer name="{layer}">
                     <SrcDataSource>PG:{pgcred}</SrcDataSource>
                     <SrcSQL>{sql}</SrcSQL>
                   {fieldremap}
                   </OGRVRTLayer>
                 </OGRVRTDataSource>
              """.format(
            layer=outlayer,
            sql=escape(sql.replace("\n", " ")),
            pgcred=pgcred,
            fieldremap=field_remap_xml,
        )
        vrtpath = os.path.join(tempfile.gettempdir(), filename + ".vrt")
        if os.path.exists(vrtpath):
            os.remove(vrtpath)
        with open(vrtpath, "w") as vrtfile:
            vrtfile.write(vrt)
        # GeoJSON writes to EPSG:4326
        if driver == 'GeoJSON' and not t_srs:
            t_srs = "EPSG:4326"
        # otherwise, default to BC Albers
        else:
            t_srs = "EPSG:3005"
        command = [
            "ogr2ogr",
            "-s_srs",
            s_srs,
            "-t_srs",
            t_srs,
            "-progress",
            "-f",
            driver,
            outfile,
            vrtpath
        ]
        # if writing to gdb, specify geom type
        if driver == "FileGDB":
            command.insert(
                len(command),
                "-nlt"
            )
            command.insert(
                len(command),
                geom_type
            )
        # automatically update existing multilayer outputs
        if driver in ("FileGDB", "GPKG") and os.path.exists(outfile):
            command.insert(
                len(command),
                "-update"
            )
        # if specified, append to existing output
        if append:
            command.insert(
                len(command),
                "-append"
            )
        subprocess.run(command)