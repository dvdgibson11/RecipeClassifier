import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

with open('klikk-recipes.json') as f:
	data = json.load(f)

train, test = train_test_split(data, train_size = 0.9)

lst = []
ingreds = {}
count = 0
countries = {}
countrycount = 0


for recipe in train:
	if len(recipe['ingredients']) == 0 or 'country' not in recipe:
		continue
	for ingred in recipe['ingredients']:
		if ingred not in ingreds:
			ingreds[ingred] = count
			count += 1
		if recipe['country'] not in countries:
			countries[recipe['country']] = countrycount
			countrycount += 1

for recipe in test:
	if len(recipe['ingredients']) == 0 or 'country' not in recipe:
		continue
	for ingred in recipe['ingredients']:
		if ingred not in ingreds:
			ingreds[ingred] = count
			count += 1
		if recipe['country'] not in countries:
			countries[recipe['country']] = countrycount
			countrycount += 1

formattedxtest = []
formattedxtrain = []
formattedytest = []
formattedytrain = []

for recipe in train:
	if len(recipe['ingredients']) == 0 or 'country' not in recipe:
		continue
	line = [0] * count
	for ingred in recipe['ingredients']:
		line[ingreds[ingred]] = 1
	formattedxtrain.append(line)
	formattedytrain.append(countries[recipe['country']])

for recipe in train:
	if len(recipe['ingredients']) == 0 or 'country' not in recipe:
		continue
	line = [0] * count
	for ingred in recipe['ingredients']:
		line[ingreds[ingred]] = 1
	formattedxtest.append(line)
	formattedytest.append(countries[recipe['country']])

LR = LogisticRegression()
LR.fit(formattedxtrain, formattedytrain)
preds = LR.predict(formattedxtest)

correct = 0
for i in range(len(preds)):
	if preds[i] == formattedytest[i]:
		correct += 1

print ('percent is', correct / len(preds))
