def format_dapi_score(cls, meta, offset):
        '''Format the line with DAPI user rating and number of votes'''
        if 'average_rank' and 'rank_count' in meta:
            label = (cls._nice_strings['average_rank'] + ':').ljust(offset + 2)
            score = cls._format_field(meta['average_rank'])
            votes = ' ({num} votes)'.format(num=meta['rank_count'])
            return label + score + votes
        else:
            return ''