from CARE import CARE, Patient
import time

start_time = time.time()
file1 = 'file1.csv'
file2 = 'file2.csv'
file3 = 'file3.csv'
patient_zero = Patient('1', '0', '60', ['27509', '30000', 'V700'], '05/31/1994')

careobj = CARE(file1)
#careobj.predict(patient_zero, 'CARE')
#print(careobj.accuracy('ICARE'))

print('--- %s seconds ---' %(time.time() - start_time))