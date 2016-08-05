# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: CARE.py
Project: My Health Prognosticator
Team: Team 5k
Org: Optum Technology Development Program Summer 2016
Creator: Jason Ko
Date: July 27 2016
Last Updated: August 5 2016
Related files:
    $dxref 2015.csv - This is a file provided by HCUP, allows us to identify icd9 codes
    				  with a readable description of the disease

    file.csv		- This is the patient history file you want to parse. 
    				  Should contain the following columns:
    				  { Member System ID, Adjudication Date, Patient Birth Date,
						Patient Gender Code, Diagnosis One Code,
						Diagnosis Two Code, Diagnosis Three Code,
						Diagnosis Four Code, Diagnosis Five Code }
Description:
        This is a python library that projects potential future diseases of a patient based on the 
        past histories of other patients. Based on the research paper "Time to CARE: 
        a collaborative engine for practical disease prediction"

How to import library:

	from CARE import CARE, Patient

Initialization steps:
- Must define file name and place files in correct folder
- The '$dxref 2015.csv' file must be in the same folder as this program
- Create a target patient to make predictions on

"""




# ------------------------------- IMPORTS ---------------------------------#

import numpy as np
import pandas as pd
import time
import math
import sys
import csv
from datetime import date, datetime


# ------------------------------- CLASSES ---------------------------------#

class Visit:
	
	"""Holds information about a visit for each patient
	
	MEMBER VARIABLES
	visit - list of disease codes
	adj_date - The adjudication date of the visit

	FUNCTIONS
	getVisit() - returns the list of disease codes
	getDate() - return the adjudication date for this visit

	"""


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

	FUNCTIONS
	getMemID() - returns the member id of the patient
	getGender() - returns the gender of the patient
	getAge() - returns the age of the patient
	getVisits() - returns the list of visit objects
	getUnique() - returns the set of disease codes by the patient
	addVisit(adj_date, visit) - adds a visit object to the list of visits

	"""

	def __init__(self, mem_id, gender, age, visit, adj_date):
		self.visits = []
		self.mem_id = mem_id
		self.gender = gender
		self.age = age
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


class CARE:
	
	"""The CARE library. The meat of the program that analyzes 
	   and generates predictions about future diseases a patient may develop, 
	   based on the experiences of other similar patients.

	   There are 3 methods of the CARE framework. They are:

	   		CARE - The basic implementation of collaborative filtering

	   		ICARE - Iterative CARE that places patients into individual disease groups and applies
	   				collaborative filtering on each disease group and aggregates the results

	   		Time-sensitive ICARE - A time sensitive ICARE system which exploits the temporal pattern in
	   							   which diseases occur, using the length of time between patient visits.


		MEMBER VARIABLES
		patients - List of patients in the form of a dictionary { patient mem_id : patient object }
		diseases - List of diseases with patients that have that disease {icd9 code : list of patients that have that disease }
		disease_codes - List of all disease codes
		dic - Dictionary that maps the icd9 code with a readable description of that code


		FUNCTIONS
		getPatients() - returns the list of patients
		getDiseases() - returns the list of diseases
		getDiseaseCodes() - returns the list of disease codes
		getDic() - returns the dictionary

		setupCARE(file) 
			- Sets up CARE by parsing the patient data file, and creating the patient
			  and disease database for prediction analysis

		train(target_patient) 
			- Trains the database based on the target patient, returns the list of the trained patient data
				(training involves filtering out patients that 
				 have less than 2 common diseases with the target patient)

		evaluate(target_patient, patient_set, disease_set, mode) 
			- evaluates and applies the desired method onto the data, and returns a list of
			  future predicted diseases for the patient
			- The three modes are: 'CARE', 'ICARE', 'TIME_ICARE'

		predict(target_patient, mode)
			- A helper function that calls evaluate and ranks the list
			  of predicted diseases, and prints the top 10 predicted diseases
			- This method is intended to be called by the user, as it is easy and simple to use

		printPatient(patient, dictionary)
			- Outputs the list of diseases belonging to the patient

		printDiseases(patient, disease_predictions, dictionary)
			- Outputs the top 10 predicted diseases for the patient in ranked order

		**COMING SOON**
		accuracy(mode)
			- Displays the accuracy of the given dataset comparing predicted diseases with those that actually
			  happen in the patient

		USAGE
		
		1. To begin using this library, first initialize the CARE object
			eg) careObj = CARE(file)

		2. Initialze the patient that you want to make predictions on
			eg) target_patient = Patient('000001', '0', '35', ['27509', '30000', 'V700'], '05/31/2013')

		3. Finally, you can make predictions, based on the desired method
			eg) careObj.predict(target_patient, 'ICARE')


	"""

	def __init__(self, filename):
		self.patients, self.diseases, self.disease_codes, self.dic = self.setupCARE(filename)

	def getPatients(self):
		return self.patients

	def getDiseases(self):
		return self.diseases

	def getDiseaseCodes(self):
		return self.disease_codes

	def getDic(self):
		return self.dic


	##### FUNCTION TO SET UP DATA FOR ANALYSIS #####
	def setupCARE(self, filename):

		def cleanData(filename):
			
			def calculate_gender(gender):
				if gender == 'M':
					return 0
				elif gender == 'F':
					return 1
				else:
					return 2
			def calculate_age(born):
				today = date.today()
				b_date = datetime.strptime(born, '%m/%d/%Y')
				return today.year - b_date.year - ((today.month, today.day) < (b_date.month, b_date.day))


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

		def parseCSV(categoryfile='$dxref 2015.csv'):
			"""
			USAGE
			categoryfile - The file provided by HCUP. Should be called '$dxref 2015.csv'

			RETURNS
			dictionary - Dictionary mapping from {icd9 codes : icd9 description }
			"""

			dictionary = {}

			count = 0
			with open(categoryfile, 'rb') as csvfile:
				datareader = csv.reader(csvfile)

				for row in datareader:
					if count >= 3:
						row[0] = row[0].replace("'","").strip()
						dictionary[row[0]] = row[3]
					count+=1

			csvfile.close()

			return dictionary

		def createPatients(df, disease_codes):

			def parse_diags(diag_list, disease_codes):

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


				new_list = []
				for diag in diag_list:
					new_diag = check_valid(diag, disease_codes)
					(new_list.append(new_diag) if new_diag is not 0 else 0)
				return new_list

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

		df = cleanData(filename)
		dic = parseCSV()
		disease_codes = set(dic.keys())
		patients, diseases = createPatients(df, disease_codes)
		return patients, diseases, disease_codes, dic


	##### FUNCTION TO FILTER DATA FOR TRAINING SETS #####
	def train(self, target):
		patient_train = {}
		disease_train = {}
		target_diseases = target.getUnique()

		for patient in self.patients.values():
			combined = target_diseases & patient.getUnique()
			if len(combined) >= 2:
				patient_train[patient.getMemID()] = patient
				for disease in patient.getUnique():
					if disease not in disease_train:
						disease_train[disease] = set()
					disease_train[disease].add(patient.getMemID())
			

		return patient_train, disease_train


	##### COLLABORATIVE FILTERING #####
	def evaluate(self, a, patient_set, disease_set, mode):

		def w(a, i):
			
			def f(self, j):
				"""Returns: log(# of patients in database / # of patients with disease j)"""
				return np.log( (1.0)*len(patient_set) / len(disease_set[j]) )

			total_sum = 0
			combined = a.getUnique() & i.getUnique()
			for disease in combined:
				first_half = f(self, disease) / math.sqrt(sum(f(self, k)**2 for k in a.getUnique()))
				second_half = f(self, disease) / math.sqrt(sum(f(self, k)**2 for k in i.getUnique()))
				total_sum += first_half * second_half
			return total_sum

		def K(self, a):
			return 1.0 / (sum(w(a, i) for i in patient_set.values()))

		def V(self, j):
			return (1.0) * len(disease_set[j]) / len(patient_set)

		def V_C(self, j, c):
			return (1.0) * len(disease_set[j] & disease_set[c]) / len(patient_set)

		def z(j, c):
			
			def S(self, p):
				n1 = len(self.diseases[c])
				n2 = len(self.patients)
				return math.sqrt( (p * (1.0 - p) / n1) + (p * (1.0 - p) / n2) )

			p1 = V_C(self, j, c)
			p2 = V(self, j)
			weighted_avg = (p1 + p2) / 2
			score = (p1 - p2) / S(self, weighted_avg)
			return score

		def getCARE(self):
			disease_score = []
			norm_constant = K(self, a)
			for disease in disease_set.keys():
				score = V(self, disease) + norm_constant * (1.0 - V(self, disease)) * (sum(w(a, patient_set[i]) for i in disease_set[disease]))
				disease_score.append([score, disease])
			return disease_score

		def getICARE(self):
			disease_score = []
			norm_constant = K(self, a)
			for j in disease_set.keys():
				max_score = 0
				for c in a.getUnique():
					if j == c:
						continue
					if z(j, c) >= 1.96 or z(j, c) <= -1.96:
						combined = disease_set[c] & disease_set[j]
						current_score = V_C(self, j, c) + norm_constant * (1.0 - V_C(self, j, c)) * (sum(w(a, patient_set[i]) for i in combined))
						if current_score > max_score:
							max_score = current_score
				disease_score.append([max_score, j])

			return disease_score

		if mode == 'CARE':
			disease_score = getCARE(self)
		
		elif mode == 'ICARE':
			disease_score = getICARE(self)
		
		disease_score.sort(key = lambda x: x[0], reverse=True)
		return disease_score


	##### HELPER FUNCTION TO PREDICT #####
	def predict(self, target, mode):

		# Filter the data for training sets
		if mode == 'CARE':
			patient_train, disease_train = self.train(target)
		else:
			patient_train = self.patients
			disease_train = self.diseases

		disease_score = self.evaluate(target, patient_train, disease_train, mode)
		

		self.printPatient(target, self.dic)
		self.printDiseases(target, disease_score[:20], self.dic)


	##### PRINT FUNCTIONS #####
	def printPatient(self, patient, dic):
		count = 1
		print('The patient has the following diseases:')
		for disease in patient.getUnique():
			print('\t%d. ' %count + dic[disease] + ' (' + disease + ')')
			count+=1
		print('\n')

	def printDiseases(self, patient, predDisease, dic):
		count = 1
		print('The patient has a possibility of getting the following diseases:')
		for disease in predDisease:
			if disease[1] in patient.getUnique():
				continue
			if disease[0] <= 0.0:
				break
			print('\t%d. ' %count + dic[disease[1]] + ' (' + disease[1] + \
				  ') -- ' + '{0:.2f}'.format(disease[0]))
			count+=1
			if count == 11:
				break
		print('\n')


	##### EVALUATION FUNCTIONS #####

	def accuracy(self, mode):

		def p(k):
			return 2.0**((-1.0 * k)/a)

		def delta(i, k, R):
			if R[k][1] in i.getUnique():
				return 1.0
			else:
				return 0.0

		N = self.patients.values()[0:30]
		a = 5
		first_part = 100.0/len(N)
		second_part = 0

		for patient in N:
			
			if mode == 'CARE':
				patient_train, disease_train = self.train(patient)
			else:
				patient_train = self.patients
				disease_train = self.diseases
			if len(patient_train) == 0 or len(disease_train) == 0:
				continue

			R = self.evaluate(patient, patient_train, disease_train, mode)
			M = []
			for index in range(len(R)):
				if delta(patient, index, R) == 1:
					M.append(R[index][1])


			numerator = sum( (p(k) * delta(patient, k, R)) for k in range(len(R)) )
			denominator = sum( p(k) for k in range(len(M)) )
			second_part += numerator/denominator

		return first_part * second_part

