def add_modality(output_path, modality):
    """Modality can be appended to the file name (such as 'bold') or use in the
    folder (such as "func"). You should always use the specific modality ('bold').
    This function converts it to the folder name.
    """
    if modality is None:
        return output_path
    else:
        if modality in ('T1w', 'T2star', 'FLAIR', 'PD'):
            modality = 'anat'

        elif modality == 'bold':
            modality = 'func'

        elif modality == 'epi':  # topup
            modality = 'fmap'

        elif modality in ('electrodes', 'coordsystem', 'channels'):
            modality = 'ieeg'

        elif modality == 'events':
            raise ValueError('modality "events" is ambiguous (can be in folder "ieeg" or "func"). Assuming "ieeg"')

        return output_path / modality