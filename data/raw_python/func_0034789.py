def to_json(self, schema_info=True):
        """ JSON serialization method that account for all information-wise important part of breakpoint graph
        """
        genomes = set()
        result = {}
        result["edges"] = []
        for bgedge in self.edges():
            genomes |= bgedge.multicolor.colors
            result["edges"].append(bgedge.to_json(schema_info=schema_info))
        result["vertices"] = [bgvertex.to_json(schema_info=schema_info) for bgvertex in self.nodes()]
        result["genomes"] = [bggenome.to_json(schema_info=schema_info) for bggenome in genomes]
        return result