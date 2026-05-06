def askInitial():
    '''Asks the user for what it wants the script to do

    Returns:
        [dictionary] -- answers to the questions
    '''
    return inquirer.prompt([
        inquirer.Text(
            'inputPath', message="What's the path of your input file (eg input.csv)"),
        inquirer.List(
            'year',
            message="What year are you in",
                    choices=[1, 2, 3, 4]
        ),
        inquirer.Checkbox(
            'whatToDo',
            message="What can I do for you (select with your spacebar)",
            choices=[
                "Get your weighted average",
                "Get your rank in the year",
                "Reformat results by module and output to csv",
                "Plot the results by module"

            ]),
    ])