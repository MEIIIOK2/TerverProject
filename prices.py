a=''
with open('риск.csv',mode='r',encoding='utf-8')as dat:
    for line in dat:
        a = set(line.split(','))
        print(a)
