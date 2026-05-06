def addMov(self, product, quantity=None, mode="buy", stop_limit=None,
               auto_margin=None, name_counter=None):
        """main function for placing movements
        stop_limit = {'gain': [mode, value], 'loss': [mode, value]}"""
        # ~ ARGS ~
        if (not isinstance(product, type('')) or
                (not isinstance(name_counter, type('')) and
                 name_counter is not None)):
            raise ValueError('product and name_counter have to be a string')
        if not isinstance(stop_limit, type({})) and stop_limit is not None:
            raise ValueError('it has to be a dictionary')
        # exclusive args
        if quantity is not None and auto_margin is not None:
            raise ValueError("quantity and auto_margin are exclusive")
        elif quantity is None and auto_margin is None:
            raise ValueError("need at least one quantity")
        # ~ MAIN ~
        # open new window
        mov = self.new_mov(product)
        mov.open()
        mov.set_mode(mode)
        # set quantity
        if quantity is not None:
            mov.set_quantity(quantity)
            # for best performance in long times
            try:
                margin = mov.get_unit_value() * quantity
            except TimeoutError:
                mov.close()
                logger.warning("market closed for %s" % mov.product)
                return False
        # auto_margin calculate quantity (how simple!)
        elif auto_margin is not None:
            unit_value = mov.get_unit_value()
            mov.set_quantity(auto_margin * unit_value)
            margin = auto_margin
        # stop limit (how can be so simple!)
        if stop_limit is not None:
            mov.set_limit('gain', stop_limit['gain'][0], stop_limit['gain'][1])
            mov.set_limit('loss', stop_limit['loss'][0], stop_limit['loss'][1])
        # confirm
        try:
            mov.confirm()
        except (exceptions.MaxQuantLimit, exceptions.MinQuantLimit) as e:
            logger.warning(e.err)
            # resolve immediately
            mov.set_quantity(e.quant)
            mov.confirm()
        except Exception:
            logger.exception('undefined error in movement confirmation')
        mov_logger.info(f"added {mov.product} movement of {mov.quantity} " +
                        f"with margin of {margin}")
        mov_logger.debug(f"stop_limit: {stop_limit}")