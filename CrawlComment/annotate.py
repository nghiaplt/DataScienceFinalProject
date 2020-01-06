filename = "encoded_angry.txt"
fileout = "final_angry.csv"
filecache = "cached_" + filename
default_label = "angry"
#### CHANGE ABOVE #####

from os import system, name
def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')


map = {}
map["neutral"] = 0
map["happy"] = 1
map["angry"] = 2
map["sad"] = 3

default_value = map[default_label]
print("If lần đầu xài thì nhớ thêm header vào file csv")
try:
    f = open(filename if int(input("Continue last work (0/1: NO/YES): ")) == 0 else filecache , "r")
except:
    print("File is not found!!!!")
    exit()
lines = f.readlines()
f.close()
val = []
label = []
i = 0
f = open(fileout,"a")
while len(lines) > 0:
    clear()
    line = lines[0][:-1]
    print("default: " + str(default_value) + '\n' + str(map) + " space to remove. '`' to stop." +'\n')
    print("remain: " + str(len(lines)))
    print(line)
    a = input()
    if len(a) == 0:
        val.append(line)
        label.append(default_value)
        f.write('{}\t{}\t\n'.format(line, default_value))
    elif a == " ":
        pass
    elif "0" <= a <= "3":
        val.append(line)
        label.append(int(a))
        f.write('{}\t{}\t\n'.format(line, int(a)))
    elif a == '`':
        f.close()
        f = open(filecache, "w")
        f.writelines(lines)
        f.close()
        exit()
    else:
        f.close()
        f = open(filecache, "w")
        f.writelines(lines)
        f.close()
        exit()
    lines.pop(0)

f.close()


