from CARE import CARE, Patient
import time

start_time = time.time()
file = '~/desktop/file1.csv'
patient_zero = Patient('1', '0', '60', ['27509', '30000', 'V700'], '05/31/1994')

careobj = CARE(file)
careobj.predict(patient_zero, 'ICARE')
#print(careobj.accuracy('ICARE'))

print('--- %s seconds ---' %(time.time() - start_time))