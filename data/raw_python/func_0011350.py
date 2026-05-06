def get_selection(available, selection, base='/scif/apps'):
    '''we compare the basename (the exp_id) of the selection and available, 
       regardless of parent directories'''

    if isinstance(selection, str):
        selection = selection.split(',')

    available = [os.path.basename(x) for x in available]
    selection = [os.path.basename(x) for x in selection]
    finalset = [x for x in selection if x in available]
    if len(finalset) == 0:
        bot.warning("No user experiments selected, providing all %s" %(len(available)))
        finalset = available
    return ["%s/%s" %(base,x) for x in finalset]