import json

with open('collection.json') as f:
	collection = json.load(f)
	count = 0
	for track in collection:
		if 'piano' in track[3]:
			if 'sad' in track[3]:
				print(track[1], track[2])
				count += 1

	print(count)
