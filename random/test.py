'''
attempt to make a counting closure
'''

# def counter(c=None):
#     if not c:
#         return 0, counter
#     else:
#         num, _ = c(c)
#         nextNum = num+1
#         def innerCounter(c=None):

#         return nextNum, numinnerCounter

#     return count, lambda c: (   count+1, 
#                                 lambda: counter(count=c()[0]+1))

def counter1():
    x = 0
    def add_one():
        nonlocal x
        x = x + 1
        return x
    return add_one

def counter2():
    x = [0]
    def add_one():
        x[0] = x[0] + 1
        return x[0]
    return add_one

c = counter1()
print(c())
print(c())
print(c())
print(c())

c = counter2()
print(c())
print(c())
print(c())
print(c())