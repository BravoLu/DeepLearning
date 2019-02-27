import sys
from collections import Counter
input_lines=open(sys.argv[1],'r').readlines()
output_file=open(sys.argv[2],'w')
id_modelList={}
id_colorList={}
id_model={}
id_color={}
for line in input_lines:
    parts=line[:-1].split(' ')
    id_modelList.setdefault(parts[1],[]).append(parts[2])
    id_colorList.setdefault(parts[1],[]).append(parts[3])
for k in id_modelList:
    id_model[k]=Counter(id_modelList[k]).most_common(1)[0][0]
    id_color[k]=Counter(id_colorList[k]).most_common(1)[0][0]
for line in input_lines:
    parts=line[:-1].split(' ')
    output_file.writelines(parts[0]+' ' + parts[1] + ' ' + id_model[parts[1]] + ' ' + id_color[parts[1]] + '\n')
output_file.close()
