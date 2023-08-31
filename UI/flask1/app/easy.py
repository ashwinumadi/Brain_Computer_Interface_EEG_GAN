import os
import pickle

cwd = os.getcwd()
cwd = cwd + "/EASY"
d = {}

Pz = []
Oz = []
P3 = []
P4 = []
PO7 = []
PO8 = []
O1 = []
O2 = []

for i in (os.listdir(cwd)):
    d[i] = []

print(d)

for image in (os.listdir(cwd)):
    cwd = cwd + '/' + image
    for i in os.listdir(cwd):
        f = open(cwd + '/' + i)
        while(True):
            a = f.readline().split()
            if(len(a) == 0):
                break
            a = list(map(float,a))
            Pz.append(a[0])
            Oz.append(a[1])
            P3.append(a[2])
            P4.append(a[3])
            PO7.append(a[4])
            PO8.append(a[5])
            O1.append(a[6])
            O2.append(a[7])
        print(i)
        f.close()
        d[image].append([Pz, Oz, P3, P4, PO7, PO8, O1, O2])
        Pz = []
        Oz = []
        P3 = []
        P4 = []
        PO7 = []
        PO8 = []
        O1 = []
        O2 = []
            
    cwd = os.getcwd()
    cwd = cwd + "/EASY"

print(len(d["Blue_Triangle"][0][0]))
dataset = "dataset.pkl"
open_file = open(dataset, "wb")
pickle.dump(d, open_file)
open_file.close()
    
'''
Pz = []
Oz = []
P3 = []
P4 = []
PO7 = []
PO8 = []
O1 = []
O2 = []

f = open("./EASY/Red_Triangle/Red_Triangle_0.easy")

while(True):
    a = f.readline().split()
    if(len(a) == 0):
        break
    a = list(map(float,a))
    Pz.append(a[0])
    Oz.append(a[1])
    P3.append(a[2])
    P4.append(a[3])
    PO7.append(a[4])
    PO8.append(a[5])
    O1.append(a[6])
    O2.append(a[7])
    
    
print(Pz)
f.close()
'''
    
    
