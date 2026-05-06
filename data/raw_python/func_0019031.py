def get_modelnames() -> List[str]:
        """Return a sorted |list| containing all application model names.

        >>> from hydpy.auxs.xmltools import XSDWriter
        >>> print(XSDWriter.get_modelnames())    # doctest: +ELLIPSIS
        [...'dam_v001', 'dam_v002', 'dam_v003', 'dam_v004', 'dam_v005',...]
        """
        return sorted(str(fn.split('.')[0])
                      for fn in os.listdir(models.__path__[0])
                      if (fn.endswith('.py') and (fn != '__init__.py')))