import json
texte = open("./medium.json","r").read();
array = json.loads(texte)
print(len(array['tasks']))

class Env:
    
