try:
	import rtmidi
except:
	print "rtmidi not found, no MIDI support available."

import time

from isobar.note import *

MIDIIN_DEFAULT = "IAC Driver A"
MIDIOUT_DEFAULT = "IAC Driver A"

import logging

log = logging.getLogger(__name__)


class MidiIn:
	def __init__(self, target=MIDIOUT_DEFAULT):
		self.midi = rtmidi.MidiIn()
		
		# ------------------------------------------------------------------------
		# don't ignore MIDI clock messages (is on by default)
		# ------------------------------------------------------------------------
		self.midi.ignore_types(timing=False)
		self.clock_target = None
		
		ports = self.midi.get_ports()
		if len(ports) == 0:
			raise Exception, "No MIDI output ports found"
		
		for index, name in enumerate(ports):
			if name == target:
				log.info("Found MIDI input (%s)", name)
				self.midi.open_port(index)
		
		if self.midi is None:
			log.warn("Could not find MIDI source %s, using default", target)
			self.midi.open_port(0)
	
	def callback(self, message, timestamp):
		message = message[0]
		data_type = message[0]
		
		if data_type == 248:
			if self.clock_target is not None:
				self.clock_target.tick()
		
		elif data_type == 250:
			if self.clock_target is not None:
				self.clock_target.reset_to_beat()
	
	def run(self):
		self.midi.set_callback(self.callback)
		while True:
			time.sleep(0.1)
	
	def poll(self):
		""" used in markov-learner -- can we refactor? """
		message = self.midi.get_message()
		if not message:
			return
		
		print message
		data_type, data_note, data_vel = message[0]
		
		if (data_type & 0x90) > 0 and data_vel > 0:
			# note on
			return Note(data_note, data_vel)
	
	def close(self):
		del self.midi


class MidiOut:
	def __init__(self, target=MIDIOUT_DEFAULT):
		self.midi = rtmidi.MidiOut()
		
		ports = self.midi.get_ports()
		if len(ports) == 0:
			raise Exception, "No MIDI output ports found"
		
		for index, name in enumerate(ports):
			if name == target:
				log.info("Found MIDI output (%s)" % name)
				self.midi.open_port(index)
		
		if self.midi is None:
			log.warn("Could not find MIDI target %s, using default" % target)
			self.midi.open_port(0)
	
	def tick(self, tick_length):
		pass
	
	def note_on(self, note=60, velocity=64, channel=0):
		log.debug("[midi] channel %d, note_on: (%d, %d)" % (channel, note, velocity))
		self.midi.send_message([0x90 + channel, int(note), int(velocity)])
	
	def note_off(self, note=60, channel=0):
		log.debug("[midi] channel %d, note_off: %d" % (channel, note))
		self.midi.send_message([0x80 + channel, int(note), 0])
	
	def all_notes_off(self, channel=0):
		log.debug("[midi] channel %d, all_notes_off")
		for n in range(128):
			self.note_off(n, channel)
	
	def control(self, control=0, value=0, channel=0):
		log.debug("[midi] channel %d, control %d: %d" % (channel, control, value))
		self.midi.send_message([0xB0 + channel, int(control), int(value)])
	
	def __destroy__(self):
		del self.midi
