'''
POLICY FORM
    state: {
        action:[(probability, newState, reward, terminal),
                (probability, newState, reward, terminal),
                (probability, newState, reward, terminal)],
        action:[(probability, newState, reward, terminal)]
        action:[(probability, newState, reward, terminal)]
        action:[(probability, newState, reward, terminal)]
    state: {
        action:[(probability, newState, reward, terminal)],
        action:[(probability, newState, reward, terminal)]
        action:[(probability, newState, reward, terminal)]
        action:[(probability, newState, reward, terminal)]
    }

policies should add up to 1
'''

banditWalk = {
    0:{
        0:[(1.0, 0, 0.0, True)],
        1:[(1.0, 0, 0.0, True)]
    },
    1: {
        0:[(1.0, 0, 0.0, True)],
        1:[(1.0, 2, 1.0, True)]
    },
    2: {
        0:[(1.0, 2, 0.0, True)],
        1:[(1.0, 2, 0.0, True)]
    }
}