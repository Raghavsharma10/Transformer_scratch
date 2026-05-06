def from_dict(cls, data: Dict[str, Union[str, int]]):
        """Generate an `ExpGene` object from a dictionary.

        Parameters
        ----------
        data : dict
            A dictionary with keys corresponding to attribute names.
            Attributes with missing keys will be assigned `None`.

        Returns
        -------
        `ExpGene`
            The gene.
        """
        assert isinstance(data, dict)

        if 'ensembl_id' not in data:
            raise ValueError('An "ensembl_id" key is missing!')

        # make a copy
        data = dict(data)
        
        for attr in ['name', 'chromosome', 'position', 'length',
                     'type', 'source']:
            if attr in data and data[attr] == '':
                data[attr] = None            

        data['type_'] = data['type']
        del data['type']

        return cls(**data)