def ogr2pg(
        self,
        in_file,
        in_layer=None,
        out_layer=None,
        schema="public",
        s_srs=None,
        t_srs="EPSG:3005",
        sql=None,
        dim=2,
        cmd_only=False,
        index=True
    ):
        """
        Load a layer to provided pgdata database connection using OGR2OGR

        -sql option is like an ESRI where_clause or the ogr2ogr -where option,
        but to increase flexibility, it is in SQLITE dialect:
        SELECT * FROM <in_layer> WHERE <sql>
        """
        # if not provided a layer name, use the name of the input file
        if not in_layer:
            in_layer = os.path.splitext(os.path.basename(in_file))[0]
        if not out_layer:
            out_layer = in_layer.lower()
        command = [
            "ogr2ogr",
            "-t_srs",
            t_srs,
            "-f",
            "PostgreSQL",
            "PG:host={h} user={u} dbname={db} password={pwd}".format(
                h=self.host, u=self.user, db=self.database, pwd=self.password
            ),
            "-lco",
            "OVERWRITE=YES",
            "-overwrite",
            "-lco",
            "SCHEMA={schema}".format(schema=schema),
            "-lco",
            "GEOMETRY_NAME=geom",
            "-dim",
            "{d}".format(d=dim),
            "-nlt",
            "PROMOTE_TO_MULTI",
            "-nln",
            out_layer,
            in_file
        ]
        if sql:
            command.insert(
                len(command),
                "-sql"
            )
            command.insert(
                len(command),
                "SELECT * FROM {} WHERE {}".format(in_layer, sql)
            )
            command.insert(len(command), "-dialect")
            command.insert(len(command), "SQLITE")
        # only add output layer name if sql not included (it gets ignored)
        if not sql:
            command.insert(
                len(command),
                in_layer
            )
        if s_srs:
            command.insert(len(command), "-s_srs")
            command.insert(len(command), s_srs)
        if not index:
            command.insert(len(command), "-lco")
            command.insert(len(command), "SPATIAL_INDEX=NO")
        if cmd_only:
            return " ".join(command)
        else:
            subprocess.run(command)