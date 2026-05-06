def import_data( self, raw_buffer ):
        """Import data from a byte array.

        raw_buffer
            Byte array to import from.
        """
        klass = self.__class__
        if raw_buffer:
            assert common.is_bytes( raw_buffer )
#            raw_buffer = memoryview( raw_buffer )

        self._field_data = {}

        for name in klass._fields:
            if raw_buffer:
                self._field_data[name] = klass._fields[name].get_from_buffer(
                    raw_buffer, parent=self
                )
            else:
                self._field_data[name] = klass._fields[name].default
        
        if raw_buffer:
            for name, check in klass._checks.items():
                check.check_buffer( raw_buffer, parent=self )

            # if we have debug logging on, check the roundtrip works
            if logger.isEnabledFor( logging.INFO ):
                test = self.export_data()
                if logger.getEffectiveLevel() <= logging.DEBUG:
                    logger.debug( 'Stats for {}:'.format( self ) )
                    logger.debug( 'Import buffer size: {}'.format( len( raw_buffer ) ) )
                    logger.debug( 'Export size: {}'.format( len( test ) ) )
                    if test == raw_buffer:
                        logger.debug( 'Content: exact match!' )
                    elif test == raw_buffer[:len( test )]:
                        logger.debug( 'Content: partial match!' )
                    else:
                        logger.debug( 'Content: different!' )
                        for x in utils.hexdump_diff_iter( raw_buffer[:len( test )], test ):
                            logger.debug( x )
                elif test != raw_buffer[:len( test )]:
                    logger.info( '{} export produced changed output from import'.format( self ) )

#        if raw_buffer:
#            raw_buffer.release()
        return