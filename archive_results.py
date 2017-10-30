def archive_results(exp_id, ga):
    """
    Script that archives the results of the experiment into a separate folder
    exp_id - a text id that will be used to name the target folder
    ga - the GeneticAlgorithm object instance to be saved
    """
    # Imports
    import os
    import pickle
    import shutil

    script_path = os.getcwd()   
    exp_path = script_path + '\\' + exp_id
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    # Pickle the champions for future analysis:    
    pickle.dump( ga.champions, open( "champions.p", "wb" ) )
    # Pickle the genetic algorithm object to save all of the setting used:
    pickle.dump( ga, open( "ga.p", "wb") )

    # to load pickle (need to go to the experiment folder):
    # champions = pickle.load( open( "champions.p", "rb" ) )    

    ### Throw all of the intermidiate files into the experiment folder
    source = os.listdir(script_path)
#    import shutil
    for files in source:
        if files.endswith(".txt") | files.endswith(".png") | files.endswith(".p"):
            shutil.move(files,exp_path)
        if files.endswith(".py"):
            shutil.copy(files,exp_path)