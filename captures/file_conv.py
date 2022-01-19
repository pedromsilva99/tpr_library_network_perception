import argparse
import numpy as np
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='?',required=True, help='input file')
    parser.add_argument('-o', '--output', nargs='?',required=True, help='output file')
    args=parser.parse_args()
    
    fileInput=args.input
    data=np.loadtxt(fileInput,dtype=int)


    fi = open(args.output, "w")
    
    for i in data:
        fi.write(str(i[2]) + " " + str(i[4]) + "\n") #upload bytes / download bytes
                                                     #i[1], i[3] para packet length
    fi.close()

if __name__ == '__main__':
    main()
