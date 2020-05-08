the idea for lunarlandar_c_dql.py:

a continuout version of dql
so not like actor critic, because that isnt value based
i want to make dql output 2 parameters for each movement dimension
then multiply its logprob into a value estimation 
    (since we are making the samples not be the values)
then after that just use temporal difference exactly like in dql
the idea is once you have that, then apply the dueling and double and nstep
modifications to it.
dueling double dql kicks ass so i jsut wanted to use that algo for continuous spaces
because i feel that dueling ddql is already a bitchin tool and i wanna apply it to more 
real life problems

working out lining up the numpy arraays and whatnot has taken about 3 days, 
and im not terribly confident the algorithm will work in the first place

so im abandoning this for now, knowing i can learn more quickly by implementing other 
algorithms that are known good in drl.

im sorry to this code for being left this way.
it actually makes me sad to leave it unfinished. really...

but maybe this is a smart decision.
or maybe im abandoning an entire tree of algorithms that function well
(however i suspect there are many drl algos that can emulate human level intelligence)

im moving forward to other continous action space models, 
and then hopefully an on-model algo next

goodnight sweet prince my code.
~chuuu