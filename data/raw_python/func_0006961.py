def _get_visuals(user):
    """
    Renvoi les éléments graphiques d'un utilisateur.

    :param user: Dictionnaire d'infos de l'utilisateur
    :return QPixmap,QLabel: Image et nom
    """
    pixmap = SuperUserAvatar() if user["status"] == "admin" else UserAvatar()
    label = user["label"]
    return pixmap, QLabel(label)