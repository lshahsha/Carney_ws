import numpy as np
import pandas as pd 

data_dir = "/Users/lshahsha/Documents/GitHub/Carney_ws/data"

def clean_data(task_name = "interference"):
    """gets the columns to be used for modelling and saves a new version of the data 

    Args:
        task_name (str, optional): name of the task. Defaults to "interference".
    """
    # load the data for the task
    task_file = f"{data_dir}/{task_name}.csv"
    df = pd.read_csv(task_file, sep=",")
    # get how many subjects we have
    print(f"number of subjects for {task_name} task: {len(df.subj_id.unique())}")
    # get a list of subjects
    subj_list = df.subj_id.unique()

    # key presses column contains data for the time of key presses, what key was pressed and whether it was correct
    # this column contains a list containing a dictionary for each key press
    # keys of this dictionary are 'correct', 'key', 'time'
    # we now get those values in separate columns

    # first step is to get the dictionary out of the list
    # the list is stored as a string, so we need to convert it to a list first
    keyp_col = df['key_presses'].apply(lambda x: eval(x))

    # now we can get the values of the dictionary
    # if the list is empty, we will get None
    keyp_col = keyp_col.apply(lambda x: x[0] if len(x) > 0 else None)

    # now we can get the values of the dictionary
    # first we get the 'correct'! if the dictionary is None, we will set it to False
    df['correct'] = keyp_col.apply(lambda x: x['correct'] if x is not None else False)

    # now we get the key pressed
    # if the dictionary is None, we will set it to None
    df['key'] = keyp_col.apply(lambda x: x['key'] if x is not None else None)

    # now we get the time of the key press
    # if the dictionary is None, we will set it to None
    df['keyp_time'] = keyp_col.apply(lambda x: x['time'] if x is not None else None)

    # save the new dataframe in data_dir
    # to keep the original, save it with a different name: <task_name>_cleaned.csv
    df.to_csv(f"{data_dir}/{task_name}_cleaned.csv", index=False, sep =  ",")

    return df



if __name__ == "__main__":

    # list of tasks 
    tasks = ["interference", "prp_and_single", "random", "switching_coherence", "switching_interference"]

    # loop over tasks, clean the data and save it
    for task in tasks:
        print(f"cleaning data for {task}")
        clean_data(task)
    pass