def get_software_package_compilation_timestamp(cls,calc,**kwargs):
        """
        Returns the timestamp of package/program compilation in ISO 8601
        format.
        """
        from dateutil.parser import parse
        try:
            date = calc.out.job_info.get_dict()['compiled']
            return parse(date.replace('_', ' ')).isoformat()
        except Exception:
            return None