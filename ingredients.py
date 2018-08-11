import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

with open('klikk-recipes.json') as f:
	data = json.load(f)

# minimum number of recipe ingredients
ingredscutoff = 3

lst = []
ingreds = {}
count = 0
countries = {}
countrycount = 0

# map each ingredient and country to a number for later binary encoding
for recipe in data:
	if len(recipe['ingredients']) < ingredscutoff or 'country' not in recipe:
		continue
	for ingred in recipe['ingredients']:
		if ingred not in ingreds:
			ingreds[ingred] = count
			count += 1
		if recipe['country'] not in countries:
			countries[recipe['country']] = countrycount
			countrycount += 1

# perform actual binary encoding of the data
formattedx = []
formattedy = []
for recipe in data:
	if len(recipe['ingredients']) < ingredscutoff or 'country' not in recipe:
		continue
	line = [0] * count
	for ingred in recipe['ingredients']:
		line[ingreds[ingred]] = 1
	formattedx.append(line)
	formattedy.append(countries[recipe['country']])

Xtrain, Xtest, ytrain, ytest = train_test_split(formattedx, formattedy, train_size = 0.8)

LR = LogisticRegression()
LR.fit(Xtrain, ytrain)
preds = LR.predict(Xtest)

correct = 0
for i in range(len(preds)):
	if preds[i] == ytest[i]:
		correct += 1

print ('logistic regression percent is', correct / len(preds))