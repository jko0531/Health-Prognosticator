from CARE import CARE, Patient
import time

start_time = time.time()
file = '~/desktop/file1.csv'
<<<<<<< HEAD
patient_one = Patient('2', '1', '60', ['7575', '71956', '7829'], '06/13/2012')

careobj = CARE(file)
careobj.predict(patient_one, 'CARE')
=======
patient_zero = Patient('1', '0', '60', ['27509', '30000', 'V700'], '05/31/1994')

careobj = CARE(file)
careobj.predict(patient_zero, 'ICARE')
#print(careobj.accuracy('ICARE'))
>>>>>>> edb8cad17596c2fbdbf65c461df64765c33fb1ce

print('--- %s seconds ---' %(time.time() - start_time))