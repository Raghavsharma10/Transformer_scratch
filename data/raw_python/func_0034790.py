def from_json(cls, data, genomes_data=None, genomes_deserialization_required=True, merge=False):
        """ A JSON deserialization operation, that recovers a breakpoint graph from its JSON representation

          as information about genomes, that are encoded in breakpoint graph might be available somewhere else, but not the
          json object, there is an option to provide it and omit encoding information about genomes.
        """
        result = cls()
        merge = merge
        vertices_dict = {}
        genomes_dict = genomes_data if genomes_data is not None and not genomes_deserialization_required else None
        if genomes_dict is None:
            ############################################################################################################
            #
            # if we need to recover genomes information from breakpoint graph json object
            # we are happy to do that
            #
            ############################################################################################################
            genomes_dict = {}
            try:
                source = genomes_data if genomes_data is not None and genomes_deserialization_required else data[
                    "genomes"]
            except KeyError as exc:
                raise ValueError("Error during breakpoint graph deserialization. No \"genomes\" information found")
            for g_dict in source:
                ############################################################################################################
                #
                # if explicitly specified in genome json object, it can be decoded using provided schema name,
                # of course a decoding breakpoint graph object shall be aware of such scheme
                # (it has to be specified in the `genomes_json_schemas` class wide dict)
                #
                ############################################################################################################
                schema_name = g_dict.get(BGGenome_JSON_SCHEMA_JSON_KEY, None)
                schema_class = None if schema_name is None else cls.genomes_json_schemas.get(schema_name, None)
                genomes_dict[g_dict["g_id"]] = BGGenome.from_json(data=g_dict, json_schema_class=schema_class)
        if "vertices" not in data:
            ############################################################################################################
            #
            # breakpoint graph can not be decoded without having information about vertices explicitly
            # as vertices are referenced in edges object, rather than explicitly provided
            #
            ############################################################################################################
            raise ValueError(
                "Error during breakpoint graph deserialization. \"vertices\" key is not present in json object")
        for vertex_dict in data["vertices"]:
            ############################################################################################################
            #
            # if explicitly specified in vertex json object, it can be decoded using provided schema name,
            # of course a decoding breakpoint graph object shall be aware of such scheme
            # (it has to be specified in the `vertices_json_schemas` class wide dict)
            #
            ############################################################################################################
            schema_name = vertex_dict.get(BGVertex_JSON_SCHEMA_JSON_KEY, None)
            schema_class = None if schema_name is None else cls.vertices_json_schemas.get(schema_name, None)
            try:
                ############################################################################################################
                #
                # we try to recover a specific vertex class based on its name.
                # it does not overwrite the schema based behaviour
                # but provides a correct default schema for a specific vertex type
                #
                ############################################################################################################
                vertex_class = BGVertex.get_vertex_class_from_vertex_name(vertex_dict["name"])
            except KeyError:
                vertex_class = BGVertex
            vertices_dict[vertex_dict["v_id"]] = vertex_class.from_json(data=vertex_dict,
                                                                        json_schema_class=schema_class)
        for edge_dict in data["edges"]:
            ############################################################################################################
            #
            # if explicitly specified in edge json object, it can be decoded using provided schema name,
            # of course a decoding breakpoint graph object shall be aware of such scheme
            # (it has to be specified in the `edges_json_schemas` class wide dict)
            #
            ############################################################################################################
            schema_name = edge_dict.get(BGEdge_JSON_SCHEMA_JSON_KEY, None)
            schema = None if schema_name is None else cls.edges_json_schemas.get(schema_name, None)
            edge = BGEdge.from_json(data=edge_dict, json_schema_class=schema)
            try:
                edge.vertex1 = vertices_dict[edge.vertex1]
                edge.vertex2 = vertices_dict[edge.vertex2]
            except KeyError:
                ############################################################################################################
                #
                # as edge references a pair of vertices, we must be sure respective vertices were decoded
                #
                ############################################################################################################
                raise ValueError(
                    "Error during breakpoint graph deserialization. Deserialized edge references non-present vertex")
            if len(edge.multicolor) == 0:
                ############################################################################################################
                #
                # edges with empty multicolor are not permitted in breakpoint graphs
                #
                ############################################################################################################
                raise ValueError(
                    "Error during breakpoint graph deserialization. Empty multicolor for deserialized edge")
            try:
                edge.multicolor = Multicolor(*[genomes_dict[g_id] for g_id in edge.multicolor])
            except KeyError:
                raise ValueError(
                    "Error during breakpoint graph deserialization. Deserialized edge reference non-present "
                    "genome in its multicolor")
            result.__add_bgedge(edge, merge=merge)
        return result