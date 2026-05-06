def from_str(cls, value):
        """
        Create a MOC from a str.
        
        This grammar is expressed is the `MOC IVOA <http://ivoa.net/documents/MOC/20190215/WD-MOC-1.1-20190215.pdf>`__
        specification at section 2.3.2.

        Parameters
        ----------
        value : str
            The MOC as a string following the grammar rules.
        
        Returns
        -------
        moc : `~mocpy.moc.MOC` or `~mocpy.tmoc.TimeMOC`
            The resulting MOC
        
        Examples
        --------
        >>> from mocpy import MOC
        >>> moc = MOC.from_str("2/2-25,28,29 4/0 6/")
        """
        # Import lark parser when from_str is called
        # at least one time
        from lark import Lark, Transformer
        class ParsingException(Exception):
            pass

        class TreeToJson(Transformer):
            def value(self, items):
                res = {}
                for item in items:
                    if item is not None: # Do not take into account the "sep" branches
                        res.update(item)
                return res

            def sep(self, items):
                pass

            def depthpix(self, items):
                depth = str(items[0])
                pixs_l = items[1:][0]
                return {depth: pixs_l}

            def uniq_pix(self, pix):
                if pix:
                    return [int(pix[0])]

            def range_pix(self, range_pix):
                lower_bound = int(range_pix[0])
                upper_bound = int(range_pix[1])
                return np.arange(lower_bound, upper_bound + 1, dtype=int)

            def pixs(self, items):
                ipixs = []
                for pix in items:
                    if pix is not None: # Do not take into account the "sep" branches
                        ipixs.extend(pix)
                return ipixs

        # Initialize the parser when from_str is called
        # for the first time
        if AbstractMOC.LARK_PARSER_STR is None:
            AbstractMOC.LARK_PARSER_STR = Lark(r"""
                value: depthpix (sep+ depthpix)*
                depthpix : INT "/" sep* pixs
                pixs : pix (sep+ pix)*
                pix : INT? -> uniq_pix
                    | (INT "-" INT) -> range_pix
                sep : " " | "," | "\n" | "\r"

                %import common.INT
                """, start='value')

        try:
            tree = AbstractMOC.LARK_PARSER_STR.parse(value)
        except Exception as err:
            raise ParsingException("Could not parse {0}. \n Check the grammar section 2.3.2 of http://ivoa.net/documents/MOC/20190215/WD-MOC-1.1-20190215.pdf to see the correct syntax for writing a MOC from a str".format(value))

        moc_json = TreeToJson().transform(tree)

        return cls.from_json(moc_json)