def phase_search_names(self, source, phase):
        """Search the bundle.yaml metadata file for pipeline configurations. Looks for:
        - <phase>-<source_table>
        - <phase>-<dest_table>
        - <phase>-<source_name>

        """
        search = []

        assert phase is not None

        # Create a search list of names for getting a pipline from the metadata
        if source and source.source_table_name:
            search.append(phase + '-' + source.source_table_name)

        if source and source.dest_table_name:
            search.append(phase + '-' + source.dest_table_name)

        if source:
            search.append(phase + '-' + source.name)

        search.append(phase)

        return search