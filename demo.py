from CARE import CARE, Patient
import time

start_time = time.time()
file = 'data/file1.csv'
patient_one = Patient('2', '1', '60', ['7575', '71956', '7829'], '06/13/2012')

careobj = CARE(file)
careobj.predict(patient_one, 'ICARE')

print('--- %s seconds ---' %(time.time() - start_time))