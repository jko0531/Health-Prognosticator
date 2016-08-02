import numpy as np
import pandas as pd
import time
import math
import sys
import csv
from datetime import date, datetime
from sklearn.preprocessing import LabelEncoder

def calculate_age(born):
	today = date.today()
	b_date = datetime.strptime(born, '%m/%d/%Y')
	return today.year - b_date.year - ((today.month, today.day) < (b_date.month, b_date.day))

def calculate_gender(gender):
    if gender == 'M':
        return 0
    elif gender == 'F':
        return 1
    else:
        return 2

def check_valid(code, disease_codes):

	if code == 0 or code == '-------':
		return 0

	if code in disease_codes:
		return code
	else:
		new_code = '0' + code
		if new_code in disease_codes:
			return new_code
		else:
			new_code2 = '0' + new_code
			if new_code2 in disease_codes:
				return new_code2
			else:
				return 0

def parse_diags(diag_list, disease_codes):
	new_list = []
	for diag in diag_list:
		new_diag = check_valid(diag, disease_codes)
		(new_list.append(new_diag) if new_diag is not 0 else 0)
	return new_list


class Visit:
	
	"""Holds information about a visit for each patient"""

	def __init__(self, adj_date, visit):
		self.visit = visit
		self.adj_date = adj_date

	def getVisit(self):
		return self.visit

	def getDate(self):
		return self.adj_date


class Patient:
	
	"""Holds information regarding the details of each patient
	
	MEMBER VARIABLES
	visits - list of visit objects
	mem_id - The member ID of each patient
	Gender - 0 = Male, 1 = Female
	Age - This is pretty obvious

	"""

	def __init__(self, mem_id, gender, age, visit, adj_date):
		self.visits = []
		self.mem_id = mem_id
		self.gender = gender
		self.age = age
		#v = Visit(adj_date, visit)
		self.visits.append(Visit(adj_date, visit))
		

	def getMemID(self):
		return self.mem_id

	def getGender(self):
		return self.gender

	def getAge(self):
		return self.age

	def getVisits(self):
		return self.visits

	def addVisit(self, adj_date, visit):
		self.visits.append(Visit(adj_date, visit))
		self.visits.sort(key=lambda x: x.adj_date)

	def getUnique(self):
		unique_codes = set()
		for visit in self.visits:
			unique_codes |= set(visit.getVisit())
		return unique_codes


def parseCSV(categoryfile, labelfile):
	"""
	USAGE
	categoryfile - The file provided by HCUP. Should be called '$dxref 2015.csv'
	labelfile - The file provided by HCUP. Should be called 'dxlabel 2015.csv'

	RETURNS
	dictionary - Dictionary mapping from {icd9 codes : HCUP category }
	i2d - Index to diagnosis.. { column index : ic9 diagnosis code }
	d2i - Diagnosis to index.. { ic0 diagnosis code : column index }
	
		* note * i2d and d2i includes the age and gender as the first 2 indexes
	"""

	dictionary, better_dictionary, labels, d2i, i2d = {}, {}, {}, {}, {}
	diseases = []


	# add birthday & gender to the d2i and i2d dictionaries
	d2i['Age'] = 0
	i2d[0] = 'Age'
	d2i['Gender'] = 1
	i2d[1] = 'Gender'

	# parse the diagnosis code labels
	count = 0
	with open(labelfile, 'rb') as csvfile:
		datareader = csv.reader(csvfile)

		for row in datareader:
			if count > 3:
				labels[int(row[0])] = row[1]
				#maplabels[count-4] = int(row[0])
				diseases.append(row[1])

			count+=1

	csvfile.close()

	# parse the diagnosis codes file
	count = 0
	with open(categoryfile, 'rb') as csvfile:
		datareader = csv.reader(csvfile)

		for row in datareader:
			if count >= 3:
				row[0] = row[0].replace("'","").strip()
				row[1] = row[1].replace("'", "").strip()
				dictionary[row[0]] = labels[int(row[1])]
				better_dictionary[row[0]] = row[3]
				d2i[row[0]] = count - 1
				i2d[count - 1] = row[0]
			

			count+=1

	csvfile.close()

	return dictionary, i2d, d2i, diseases, better_dictionary


def cleanData(filename):
	relevant_columns = ['Member System ID', 'Adjudication Date', 'Patient Birth Date', \
						'Patient Gender Code', 'Diagnosis One Code', \
						'Diagnosis Two Code', 'Diagnosis Three Code', \
						'Diagnosis Four Code', 'Diagnosis Five Code']

	df = pd.read_csv(filename, usecols=relevant_columns, dtype=np.str)\
			.drop_duplicates()\
			.reset_index().drop('index', axis=1).fillna(0)
	df = df[relevant_columns]
	df['Adjudication Date'] = pd.to_datetime(df['Adjudication Date'], format='%m/%d/%Y')
	df['Patient Birth Date'] = df['Patient Birth Date'].apply(calculate_age)
	df['Patient Gender Code'] = df['Patient Gender Code'].apply(calculate_gender)

	return df


def createPatients(df, disease_codes):
	patients, diseases = {}, {}

	for row in df.itertuples():
		mem_id = row[1]
		adj_date = row[2]
		age = row[3]
		gender = row[4]
		visit = parse_diags(row[5:], disease_codes)
		
		for item in visit:
			if item not in diseases:
			    diseases[item] = set()
			diseases[item].add(mem_id)

		if mem_id not in patients:
			p = Patient(mem_id, gender, age, visit, adj_date)
			patients[mem_id] = p
		else:
			patients[mem_id].addVisit(adj_date, visit)

	return patients, diseases


def setupCARE(filename):
	categoryfile = '$dxref 2015.csv'
	labelfile = 'dxlabel 2015.csv'
	df = cleanData(filename)
	dic, i2d, d2i, foo_diseases, better_dic = parseCSV(categoryfile, labelfile)
	disease_codes = set(dic.keys())
	patients, diseases = createPatients(df, disease_codes)
	return patients, diseases, disease_codes, dic, better_dic

def train_filter(target, patients, diseases):
	# filter patients that have at least 2 common diseases
	patient_train = {}
	disease_train = {}
	target_diseases = target.getUnique()

	#for disease in target_diseases:
	#	disease_train[disease] = set()
	#	disease_train[disease].add(target.getMemID())
	
	for patient in patients.values():
		#patient_diseases = patient.getUnique()
		combined = target_diseases & patient.getUnique()
		if len(combined) >= 2:
			patient_train[patient.getMemID()] = patient
			for disease in patient.getUnique():
				if disease not in disease_train:
					disease_train[disease] = set()
				disease_train[disease].add(patient.getMemID())
		
	# remove patient from train set

	return patient_train, disease_train


def implementCARE(target, patients, diseases, disease_codes):

	### VECTOR SIMILARITY ###
	def vote(patient, disease):
		if disease in patient.getUnique():
			return 1.0
		else:
			return 0.0

	def f(j):
		"""Returns: log(# of patients in database / # of patients with disease j)"""

		return np.log( (1.0)*len(patients) / len(diseases[j]) )


	# possible optimization: turn the getUnique() set into a numpy array, then do
	# array-wise multiplication
	def w(a, i):
		total_sum = 0
		combined = a.getUnique() & i.getUnique()
		for disease in combined:
			first_half = f(disease) / math.sqrt(sum(f(k)**2 for k in a.getUnique()))
			second_half = f(disease) / math.sqrt(sum(f(k)**2 for k in i.getUnique()))
			total_sum += first_half * second_half
		return total_sum


	"""def w(a, i):
		total_sum = 0
		for j in diseases.keys():
			first_half = (f(j) * vote(a, j)) / math.sqrt(sum(f(k)**2 * vote(a, k)**2 for k in a.getUnique()))
			second_half = (f(j) * vote(i, j)) / math.sqrt(sum(f(k)**2 * vote(i, k)**2 for k in i.getUnique()))
			total_sum += first_half * second_half
		return total_sum
	"""

	### PREDICTION SCORE ###
	def V(j):
		return (1.0) * len(diseases[j]) / len(patients)

	def K(a):
		return 1.0 / (sum(w(a, i) for i in patients.values()))

	def p(a, j):
		return V(j) + K(a) * (1.0 - V(j)) * (sum(w(a, patients[i]) for i in diseases[j]))

	### BEGIN PREDICTION ###
	# TODO: need to fix & filter diseases to the train_set
	#print(p(target, '6961'))
	disease_score = []
	for disease in diseases.keys():
		score = p(target, disease)
		disease_score.append([score, disease])
	
	disease_score.sort(key = lambda x: x[0], reverse=True)

	return disease_score
	#patient_one = patients['281851294']
	#print(w(target, patient_one))


def printPatient(patient, dic):
	print('The patient has the following diseases:')
	for disease in patient.getUnique():
		print('\t-' + dic[disease] + ' (' + disease + ')')

def printDiseases(patient, predDisease, dic):
	print('The patient has a possibility of getting the following 10 diseases:')
	for disease in predDisease:
		if disease[1] in patient.getUnique():
			continue
		print('\t-' + dic[disease[1]] + ' (' + disease[1] + \
			  ') -- ' + '{0:.2f}'.format(disease[0]))

def Main():
	# files for parsing
	file1 = 'file1.csv'
	file2 = 'file2.csv'
	file3 = 'file3.csv'

	# set up for CARE
	patients, diseases, disease_codes, dic, better_dic = setupCARE(file1)
	
	# target patient
	patient_zero = Patient('1', '0', '57', ['7020', '6989', '73300'], '05/31/1994')

	# filter patients for training set, and filter diseases too
	
	patient_train, disease_train = train_filter(patient_zero, patients, diseases)

	# CARE implementation
	predDisease = implementCARE(patient_zero, patient_train, disease_train, disease_codes)
	printPatient(patient_zero, better_dic)
	printDiseases(patient_zero, predDisease[:20], better_dic)

if __name__ == '__main__':
	start_time = time.time()
	Main()
	print('--- %s seconds ---' %(time.time() - start_time))
