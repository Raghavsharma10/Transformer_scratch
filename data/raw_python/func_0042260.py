def load_attributes(self):
        '''Read the variables from the VARS_MODULE_PATH'''

        try:
            vars_path = settings.VARS_MODULE_PATH
        except Exception:
            # logger.warning("*" * 55)
            logger.warning(
                " [WARNING] Using default VARS_MODULE_PATH = '{}'".format(
                    VARS_MODULE_PATH_DEFAULT))
            vars_path = VARS_MODULE_PATH_DEFAULT

        try:
            __import__(vars_path)
        except ImportError:
            logger.warning(" [WARNING] No module named '{}'".format(
                vars_path))
            logger.warning(" Please, read the docs: goo.gl/E82vkX\n".format(
                vars_path))