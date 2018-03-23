from xml_io import *


def test_musicxml_io():
	xmlscore = MusicXML()
	xmlscore.from_file('../model/test.mxl')
	phrases = list(xmlscore.phrases(config=Config(), reanalyze=False))
	transformer = XMLtoNoteSequence()

	phrase_dict = transformer.transform(phrases[1], Config)

	chord_sequence = phrase_dict['chord']
	chord_sequence.to_midi(phrase_dict['melody'], phrase_dict['name'])


if __name__ == '__main__':
	test_musicxml_io()
