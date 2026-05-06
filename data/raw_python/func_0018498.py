def save_bed(cls, query, filename=sys.stdout):
        """
        write a bed12 file of the query.
        Parameters
        ----------

        query : query
            a table or query to save to file
        filename : file
            string or filehandle to write output

        """
        out = _open(filename, 'w')
        for o in query:
            out.write(o.bed() + '\n')