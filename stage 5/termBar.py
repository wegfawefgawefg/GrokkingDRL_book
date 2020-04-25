def printBar(char="#", length=40):
    print(char*length)

def printHeader(msg, length=40):
    print()
    printBar(length=40)
    print("####\t" + msg)
    printBar(length=40)

def printSubHeader(msg, paddChar="#"):
    print()
    print(paddChar*4 + "\t" + msg + "\t" + paddChar*4)

def printSubSubHeader(msg, paddChar="="):
    print()
    print(paddChar*2 + "\t" + msg + "\t" + paddChar*2)