height = [0,1,0,2,1,0,1,3,2,1,2,1]

def brute(height):
    highest = max(height)
    highToLow = sorted(height)
    sum = 0
    for level in range(1, highest):
        firstWall = None
        lastWall = None
        #   find first black
        for i in range(0, len(height)):
            if height[i] == level:
                firstWall = i
                break
        for i in reversed(range(0, len(height))):
            if height[i] == level:
                lastWall = i
                break
        for i in range(firstWall+1, lastWall):
            if height[i] < level:
                sum += 1
    return(sum)

def sorted(height):
    length = len(height)
    highToLow = sorted(height)
    
    #   find the heights that need to be scanned
    scanHeights = []
    for i in range(length):
        

    highest = max(height)
    sum = 0
    for level in range(1, highest):
        firstWall = None
        lastWall = None
        #   find first black
        for i in range(0, len(height)):
            if height[i] == level:
                firstWall = i
                break
        for i in reversed(range(0, len(height))):
            if height[i] == level:
                lastWall = i
                break
        for i in range(firstWall+1, lastWall):
            if height[i] < level:
                sum += 1
    return(sum)
print(brute(height))
print(sorted(height))