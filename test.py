import os
filepath = '/Users/xiechunyao/Downloads/contract/testdataset/access_control/'
data=os.listdir(filepath)
for x in data:
    if x[:x.find('.')] == 'FibonacciBalance':
        print(filepath+x)