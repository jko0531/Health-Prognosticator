
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

			# parse the diagnosis codes file
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

		#categoryfile = '$dxref 2015.csv'
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



	##### COLLABORATIVE FILTERING ALGORITHMS #####

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

		def p(self, j):
				return V(self, j) + K(self, a) * (1.0 - V(self, j)) * (sum(w(a, patient_set[i]) for i in disease_set[j]))

		def getCARE(self):
			disease_score = []
			for disease in disease_set.keys():
				score = p(self, disease)
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

		return disease_score

	##############################################
	

	def predict(self, target, mode):

		# Filter the data first
		if mode == 'CARE':
			patient_train, disease_train = self.train(target)
		else:
			patient_train = self.patients
			disease_train = self.diseases

		disease_score = self.evaluate(target, patient_train, disease_train, mode)
		disease_score.sort(key = lambda x: x[0], reverse=True)

		self.printPatient(target, self.dic)
		self.printDiseases(target, disease_score[:20], self.dic)


	##############################################

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
		print('The patient has a possibility of getting the following 10 diseases:')
		for disease in predDisease:
			if disease[1] in patient.getUnique():
				continue
			print('\t%d. ' %count + dic[disease[1]] + ' (' + disease[1] + \
				  ') -- ' + '{0:.2f}'.format(disease[0]))
			count+=1
			if count == 11:
				return

	###########################
