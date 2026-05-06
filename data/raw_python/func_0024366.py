def main():
    """
    Main method
    """
    print("Creating a new game...")

    new_game = Game(Human(color.white), Human(color.black))
    result = new_game.play()

    print("Result is ", result)