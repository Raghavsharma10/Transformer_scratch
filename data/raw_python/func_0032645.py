def take_data_triggered(self, trigger, edge, stop):
        """Configures data acquisition to start on various trigger conditions.

        :param trigger: The trigger condition, either 'curve' or 'point'.

            ======= =======================================================
            Value   Description
            ======= =======================================================
            'curve' Each trigger signal starts a curve acquisition.
            'point' A point is stored for each trigger signal. The max
                    trigger frequency in this mode is 1 kHz.
            ======= =======================================================

        :param edge: Defines wether a 'rising' or 'falling' edge is interpreted
            as a trigger signal.
        :param stop: The stop condition. Valid are 'buffer', 'halt',
            'rising' and 'falling'.

            ========= ==========================================================
            Value     Description
            ========= ==========================================================
            'buffer'  Data acquisition stops when the number of point
                      specified in :attr:`~.Buffer.length` is acquired.
            'halt'    Data acquisition stops when the halt command is issued.
            'trigger' Takes data for the period of a trigger event. If edge is
                      'rising' then teh acquisition starts on the rising edge of
                      the trigger signal and stops on the falling edge and vice
                      versa
            ========= ==========================================================

        """
        param = {
            ('curve', 'rising', 'buffer'): 0,
            ('point', 'rising', 'buffer'): 1,
            ('curve', 'falling', 'buffer'): 2,
            ('point', 'falling', 'buffer'): 3,
            ('curve', 'rising', 'halt'): 4,
            ('point', 'rising', 'halt'): 5,
            ('curve', 'falling', 'halt'): 6,
            ('point', 'falling', 'halt'): 7,
            ('curve', 'rising', 'trigger'): 8,
            ('curve', 'falling', 'trigger'): 9,
        }
        self._write(('TDT', Integer), param[(trigger, edge, stop)])