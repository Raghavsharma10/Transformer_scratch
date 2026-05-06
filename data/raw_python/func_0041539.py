def main(args):
    '''main entry point of app
    
    Arguments:
        args {namespace} -- arguments provided in cli
    '''
    
    print("\nNote it's very possible that this doesn't work correctly so take what it gives with a bucketload of salt\n")

    #########################
    #                       #
    #                       #
    #         prompt        #
    #                       #
    #                       #
    #########################

    if not len(sys.argv) > 1:
        initialAnswers = askInitial()

        inputPath = pathlib.Path(initialAnswers['inputPath'])
        year = int(initialAnswers['year'])
        # create a list from every row
        badFormat = badFormater(inputPath)  # create a list from every row
        howManyCandidates = len(badFormat) - 1

        length = int(len(badFormat['Cand'])/2)
        finalReturn = []

        if "Get your rank in the year" in initialAnswers['whatToDo']:
            candidateNumber = askCandidateNumber()
            weightedAverage = myGrades(year, candidateNumber, badFormat, length)
            rank = myRank(weightedAverage, badFormat, year, length)

            if "Get your weighted average" in initialAnswers['whatToDo']:
                finalReturn.append('Your weighted average for the year is: {:.2f}%'.format(
                    weightedAverage))

            finalReturn.append('Your rank is {}th of {} ({:.2f} percentile)'.format(
                rank, howManyCandidates, (rank * 100) / howManyCandidates))
        elif "Get your weighted average" in initialAnswers['whatToDo']:
            candidateNumber = askCandidateNumber()
            weightedAverage = myGrades(year, candidateNumber, badFormat, length)
            finalReturn.append('Your weighted average for the year is: {:.2f}%'.format(
                weightedAverage))

        if "Reformat results by module and output to csv" in initialAnswers['whatToDo']:

            formatOutputPath = pathlib.Path(askFormat())

            goodFormat = goodFormater(badFormat, formatOutputPath, year, length)

            if "Plot the results by module" in initialAnswers['whatToDo']:
                howPlotAsk(goodFormat)

        elif "Plot the results by module" in initialAnswers['whatToDo']:
            goodFormat = goodFormater(badFormat, None, year, length)
            howPlotAsk(goodFormat)

        [print('\n', x) for x in finalReturn]

    #########################
    #                       #
    #          end          #
    #         prompt        #
    #                       #
    #                       #
    #########################

    #########################
    #                       #
    #                       #
    #       run with        #
    #       cli args        #
    #                       #
    #########################

    if len(sys.argv) > 1:
        if not args.input:
            inputPath = pathlib.Path(askInput())
        else:
            inputPath = pathlib.Path(args.input)
        if not args.year:
            year = int(askYear())
        else:
            year = int(args.year)

        # create a list from every row
        badFormat = badFormater(inputPath)  # create a list from every row
        howManyCandidates = len(badFormat) - 1

        length = int(len(badFormat['Cand'])/2)
        finalReturn = []

        if args.rank:
            if not args.candidate:
                candidateNumber = askCandidateNumber()
            else:
                candidateNumber = args.candidate

            weightedAverage = myGrades(year, candidateNumber, badFormat, length)
            rank = myRank(weightedAverage, badFormat, year, length)

            if args.my:
                finalReturn.append('Your weighted average for the year is: {:.2f}%'.format(
                    weightedAverage))

            finalReturn.append('Your rank is {}th of {} ({:.2f} percentile)'.format(
                rank, howManyCandidates, (rank * 100) / howManyCandidates))

        elif args.my:
            if not args.candidate:
                candidateNumber = askCandidateNumber()
            else:
                candidateNumber = args.candidate

            weightedAverage = myGrades(year, candidateNumber, badFormat, length)
            finalReturn.append('Your weighted average for the year is: {:.2f}%'.format(
                weightedAverage))

        if args.format is not None:
            formatOutputPath = pathlib.Path(args.format)
            goodFormat = goodFormater(badFormat, formatOutputPath, year, length)

            if args.plot:
                howPlotArgs(goodFormat)
        elif args.plot:
            goodFormat = goodFormater(badFormat, None, year, length)
            howPlotArgs(goodFormat)

        [print('\n', x) for x in finalReturn]

    #########################
    #                       #
    #         end           #
    #       run with        #
    #       cli args        #
    #                       #
    #########################

    print('')