import env
import json
texte = open("./tiny.json","r").read();
array = json.loads(texte)
print(array)
myenv = env.Env(array)
state = myenv.reset()
while 1:
    print('state',state)
    action = input('select action : ')
    action = int(action)
    state = myenv.step(action)
    print(myenv.machines)
    print(myenv.operators)
    print(myenv.started_time)

