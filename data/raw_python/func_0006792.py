def ansi_format( self, width=64, height=12 ):
        """Return a human readable ANSI-terminal printout of the stats.

        width
            Custom width for the graph (in characters).

        height
            Custom height for the graph (in characters).
        """
        from mrcrowbar.ansi import format_bar_graph_iter
        if (256 % width) != 0:
            raise ValueError( 'Width of the histogram must be a divisor of 256' )
        elif (width <= 0):
            raise ValueError( 'Width of the histogram must be greater than zero' )
        elif (width > 256):
            raise ValueError( 'Width of the histogram must be less than or equal to 256' )
    
        buckets = self.histogram( width )
        result = []
        for line in format_bar_graph_iter( buckets, width=width, height=height ):
            result.append( ' {}\n'.format( line ) )

        result.append( '╘'+('═'*width)+'╛\n' )
        result.append( 'entropy: {:.10f}\n'.format( self.entropy ) )
        result.append( 'samples: {}'.format( self.samples ) )
        return ''.join( result )