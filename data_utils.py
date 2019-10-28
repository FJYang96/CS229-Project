import numpy as np
import csv

def possible_values(dataset, ind):
    '''
    Go through the set and construct a set of possible values for a specific
    index
    '''
    s = set()
    for d in dataset:
        s.add(d[ind])
    return s

def load_data(csv_dir):
    data = []
    with open(csv_dir, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            data.append(line)
    return data

'''
Given a feature vector x for a patient, extract relavant information about 
the patient
'''

def extract_age(x):
    '''
    Extract the age of patient in decades.
    If age is unknown, encode as 0
    Return: int
    '''
    age = x[4]
    if age == '10 - 19':
        return 1
    elif age =='20 - 29':
        return 2
    elif age =='30 - 39':
        return 3
    elif age =='40 - 49':
        return 4
    elif age =='50 - 59':
        return 5
    elif age =='60 - 69':
        return 6
    elif age =='70 - 79':
        return 7
    elif age =='80 - 89':
        return 8
    elif age =='90+':
        return 9
    else: 
        return 0

def extract_height(x):
    '''
    Extract height of patient
    Return: float
    '''
    h = x[5]
    if h == '' or h == 'NA':
        return 166.0
    else:
        return float(h)

def extract_weight(x):
    '''
    Extract weight of patient
    Return: float
    '''
    w = x[6]
    if w == 'NA':
        return 89.1
    else:
        return float(w)

def extract_dose(x):
    '''
    Extract therapeutic dose of warfarin
    Return: float
    '''
    return float(x[34])

def extract_race(x):
    '''
    Extract the race of the patient
    Return: one hot encoding
            [Asian, White, Black/African American, Unknown]
    '''
    race = x[2]
    encoding = np.zeros(4)
    if race == 'Asian':
        encoding[0] = 1
    elif race == 'White':
        encoding[1] = 1
    elif race == 'Black or African American':
        encoding[2] = 1
    elif race == 'Unknown':
        encoding[3]  = 1
    else:
        raise NotImplementedError
    return encoding

def extract_Amio(x):
    '''
    Whether the patient is taking Amiodarone
    Return: 1 or 0
    '''
    a = x[23]
    if a == '1':
        return 1
    else:
        return 0

def extract_enzyme(x):
    '''
    Whether the patient is taking enzyme inducer
    Return: 1 or 0
    '''
    if (x[24] == '1') or(x[25] == '1') or (x[26] == '1'):
        return 1
    else:
        return 0

def extract_CYP(x):
    cyp = x[37]
    if cyp == '*1/*2':
        return np.array([1,0,0,0,0,0])
    elif cyp == '*1/*3':
        return np.array([0,1,0,0,0,0])
    elif cyp == '*2/*2':
        return np.array([0,0,1,0,0,0])
    elif cyp == '*2/*3':
        return np.array([0,0,0,1,0,0])
    elif cyp == '*3/*3':
        return np.array([0,0,0,0,1,0])
    elif cyp == 'NA':
        return np.array([0,0,0,0,0,1])
    else:
        return np.zeros(6)
    
def extract_VKO(x):
    if x[41] == 'A/A':
        return np.array([0, 1, 0])
    elif x[41] == 'A/G':
        return np.array([1, 0, 0])
    elif x[41] == 'NA':
        return np.array([0, 0, 1])
    else:
        return np.zeros(3)

def construct_feature_vector(x):
    feature = np.ones(18)
    feature[0] = extract_age(x)
    feature[1] = extract_height(x)
    feature[2] = extract_weight(x)
    feature[3:7] = extract_race(x)
    feature[7] = extract_Amio(x)
    feature[8] = extract_enzyme(x)
    feature[9:15] = extract_CYP(x)
    feature[15:18] = extract_VKO(x)
    return feature

def patient_from_feature(x):
    feature = construct_feature_vector(x)
    dose = extract_dose(x)
    return feature, dose
