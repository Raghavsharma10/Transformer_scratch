def seaborn_set(context='poster', style='white', palette='colorblind',
                font_scale=1.3, font='serif', rc=mpl_rc):
    """
    Perform `seaborn.set(**kwargs)`.

    Additional keyword arguments are passed in using this module's
    `attr:mpl_rc` attribute.
    """
    sns.set(context="poster", style="white", palette="colorblind",
            font_scale=1.3, font="serif", rc=mpl_rc)