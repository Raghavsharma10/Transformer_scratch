def save_image(filename,view='axial',type='png',hostname=None):
    '''Save currently open AFNI view ``view`` to ``filename`` using ``type`` (``png`` or ``jpeg``)'''
    driver_send("SAVE_%s %simage %s" % (type.upper(),view.lower(),filename),hostname=hostname)