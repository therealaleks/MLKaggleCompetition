import pandas as pd
import numpy as np

#script to determine similarity % between two set of results

#accuracy of champion results to determine if we should submit to kaggle
champAcc = 91.3
#estimate percentage of champ's wrong predictions that candidate will get right
upset = 0.5

candidate = pd.read_csv("results.csv")
champ = pd.read_csv("91.csv")
candidate = np.array(candidate['Label'])
champ = np.array(champ['Label'])

same = 0.0
for i in range (14000):
    if candidate[i] == champ[i]:
        same += 1.0

passingMark = 91.3 - (upset * (100-91.3))
similarity = same/14000.0

print('similarity:  ' +str(similarity))
print('passingMark: ' + str(passingMark))

print('***************************************')

if similarity <= passingMark:
    print('FAIL')
    print('DO NOT SUBMIT')
else:
    print('PASS')