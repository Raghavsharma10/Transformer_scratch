def get_player_img(player_id):
    """
    Returns the image of the player from stats.nba.com as a numpy array and
    saves the image as PNG file in the current directory.

    Parameters
    ----------
    player_id: int
        The player ID used to find the image.

    Returns
    -------
    player_img: ndarray
        The multidimensional numpy array of the player image, which matplotlib
        can plot.
    """
    url = "http://stats.nba.com/media/players/230x185/"+str(player_id)+".png"
    img_file = str(player_id) + ".png"
    pic = urlretrieve(url, img_file)
    player_img = plt.imread(pic[0])
    return player_img