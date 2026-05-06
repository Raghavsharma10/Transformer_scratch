def get_job_config(conf):
        """ Extract handler names from job_conf.xml
        """
        rval = []
        root = elementtree.parse(conf).getroot()
        for handler in root.find('handlers'):
            rval.append({'service_name' : handler.attrib['id']})
        return rval