def make_ioc(name=None,
                 description='Automatically generated IOC',
                 author='IOC_api',
                 links=None,
                 keywords=None,
                 iocid=None):
        """
        This generates all parts of an IOC, but without any definition.

        This is a helper function used by __init__.

        :param name: string, Name of the ioc
        :param description: string, description of the ioc
        :param author: string, author name/email address
        :param links: ist of tuples.  Each tuple should be in the form (rel, href, value).
        :param keywords: string.  This is normally a space delimited string of values that may be used as keywords
        :param iocid: GUID for the IOC.  This should not be specified under normal circumstances.
        :return: a tuple containing three elementTree Element objects
         The first element, the root, contains the entire IOC itself.
         The second element, the top level OR indicator, allows the user to add
          additional IndicatorItem or Indicator nodes to the IOC easily.
         The third element, the parameters node, allows the user to quickly
          parse the parameters.
        """
        root = ioc_et.make_ioc_root(iocid)
        root.append(ioc_et.make_metadata_node(name, description, author, links, keywords))
        metadata_node = root.find('metadata')
        top_level_indicator = make_indicator_node('OR')
        parameters_node = (ioc_et.make_parameters_node())
        root.append(ioc_et.make_criteria_node(top_level_indicator))
        root.append(parameters_node)
        ioc_et.set_root_lastmodified(root)
        return root, metadata_node, top_level_indicator, parameters_node