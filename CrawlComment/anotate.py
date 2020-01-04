filename = "encoded_happy.txt"
fileout = "final_happy.txt"
default_label = "happy"
######### CHANGE ABOVE ONLY #########

from os import system, name 
import pandas as pd
def clear(): 
    if name == 'nt': 
        _ = system('cls') 
    else: 
        _ = system('clear') 


map = {}
map["neutral"] = 0
map["happy"] = 1
map["angry"] = 2
map["sad"] = 4

default_value = map[default_label]


f = open(filename, "r")
lines = f.readlines()
val = []
label = []
i = 0
while len(lines) > 0:
    clear()
    line = lines[0][:-1]
    lines.pop(0)
    print("default: " + str(default_value) + '\n' + str(map) + " space to remove" +'\n')
    print(line)
    a = input()
    if len(a) == 0:
        val.append(line)
        label.append(default_value)
    elif a == " ":
        pass
    elif "0" <= a <= "3":
        val.append(line)
        label.append(int(a))
    else:
        print("ERROR")


d = {"text": val, "label": label}
df = pd.DataFrame(d)
df.to_csv(fileout, index = false)
    

