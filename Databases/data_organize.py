'''
Getting Midi/XML files and change them into clean format: [Artist] - [Name]
'''

folder = 'Wikifonia'


def find_info(mid):
	tracks = mid.tracks
	name_found = False
	artist_found = False
	for track in tracks:
		for message in track:
			if message.is_meta:
				if not name_found and (message.type == 'track_name'):
					name = message.name
					name_found = True
				if not artist_found and (message.type == 'track_name'):
					pass

# if folder == 'clean_midi':
# 	artists = os.listdir(folder)
# 	for artist in artists:
# 		if os.path.isdir(os.path.join(folder, artist)):
# 			songs = os.listdir(os.path.join(folder, artist))
# 			for song in songs:
# 				if song.endswith('.mid'):
# 					try:
# 						os.rename(os.path.join(folder, artist, song), os.path.join('Organized', artist + ' - ' + song))
# 						print('Converted ' + artist + ' - ' + song)
# 					except OSError:
# 						print('Skipping ' + artist + ' - ' + song)


# if folder == 'Wikifonia':
# 	# unzipping
# 	for track in os.listdir(folder):
# 		if track.endswith('.mxl'):
# 			zip_ref = zipfile.ZipFile(os.path.join(folder, track), 'r')
# 			os.mkdir('Extracted/'+track[:-4])
# 			zip_ref.extractall('Extracted/'+track[:-4])
# 			zip_ref.close()
#
# 	# To song names
# 	for track in os.listdir('Extracted'):
# 		if os.path.isdir('Extracted/' + track):
# 			try:
# 				os.rename(os.path.join('Extracted', track, 'musicXML.xml'),\
# 			          os.path.join('Organized', track + '.xml'))
# 			except OSError:
# 				print('Skipping ' + track)
