#!/usr/bin/env python2
#-*- coding: utf-8 -*-

# NOTE FOR WINDOWS USERS:
# You can download a "exefied" version of self game at:
# http://hi-im.laria.me/progs/tetris_py_exefied.zip
# If a DLL is missing or something like self, write an E-Mail (me@laria.me)
# or leave a comment on self gist.

# Very simple tetris implementation
# 
# Control keys:
#       Down - Drop stone faster
# Left/Right - Move stone
#         Up - Rotate Stone clockwise
#     Escape - Quit game
#          P - Pause game
#     Return - Instant drop
#
# Have fun!

# Copyright (c) 2010 "Laria Carolin Chabowski"<me@laria.me>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of self software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and self permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


#-- for debug in windows
import os
clear = lambda: os.system('cls')

from copy import deepcopy
from random import randrange as rand
from random import randint
from random import random
import pygame, sys
import tensorflow as tf

# The configuration
cell_size =	18
cols =		10
rows =		22
maxfps = 	30
play_game = False;
train = True;


colors = [
(0,   0,   0  ),
(255, 85,  85),
(100, 200, 115),
(120, 108, 245),
(255, 140, 50 ),
(50,  120, 52 ),
(146, 202, 73 ),
(150, 161, 218 ),
(35,  35,  35) # Helper color for background grid
]

# Define the shapes of the single parts
tetris_shapes = [
	[[1, 1, 1],
	 [0, 1, 0]],
	
	[[0, 2, 2],
	 [2, 2, 0]],
	
	[[3, 3, 0],
	 [0, 3, 3]],
	
	[[4, 0, 0],
	 [4, 4, 4]],
	
	[[0, 0, 5],
	 [5, 5, 5]],
	
	[[6, 6, 6, 6]],
	
	[[7, 7],
	 [7, 7]]
]

def rotate_clockwise(shape):
	return [ [ shape[y][x]
			for y in range(len(shape)) ]
		for x in range(len(shape[0]) - 1, -1, -1) ]

def check_collision(board, shape, offset):
	off_x, off_y = offset
	for cy, row in enumerate(shape):
		for cx, cell in enumerate(row):
			try:
				if cell and board[ cy + off_y ][ cx + off_x ]:
					return True
			except IndexError:
				return True
	return False

def remove_row(board, row):
	del board[row]
	return [[0 for i in range(cols)]] + board
	
def join_matrixes(mat1, mat2, mat2_off):
	off_x, off_y = mat2_off
	for cy, row in enumerate(mat2):
		for cx, val in enumerate(row):
			mat1[cy+off_y-1	][cx+off_x] += val
	return mat1

def new_board():
	board = [ [ 0 for x in range(cols) ]
			for y in range(rows) ]
	board += [[ 1 for x in range(cols)]]
	return board

class TetrisApp(object):
	def __init__(self):
		pygame.init()
		pygame.key.set_repeat(250,25)
		self.width = cell_size*(cols+5)
		self.height = cell_size*rows
		self.rlim = cell_size*cols
		self.bground_grid = [[ 8 if x%2==y%2 else 0 for x in range(cols)] for y in range(rows)]
		
		self.default_font =  pygame.font.Font(
			pygame.font.get_default_font(), 12)
		
		self.screen = pygame.display.set_mode((self.width, self.height))
		pygame.event.set_blocked(pygame.MOUSEMOTION) # We do not need
		                                             # mouse movement
		                                             # events, so we
		                                             # block them.
		self.next_stone = tetris_shapes[rand(len(tetris_shapes))]
		self.init_game()
	
	def new_stone(self):
		self.stone = self.next_stone[:]
		self.next_stone = tetris_shapes[rand(len(tetris_shapes))]
		self.stone_x = int(cols / 2 - len(self.stone[0])/2)
		self.stone_y = 0
		
		if check_collision(self.board,
		                   self.stone,
		                   (self.stone_x, self.stone_y)):
			self.gameover = True
	
	def init_game(self):
		self.board = new_board()
		self.new_stone()
		self.level = 1
		self.score = 0
		self.lines = 0
		pygame.time.set_timer(pygame.USEREVENT+1, 1000)
	
	def disp_msg(self, msg, topleft):
		x,y = topleft
		for line in msg.splitlines():
			self.screen.blit(
				self.default_font.render(
					line,
					False,
					(255,255,255),
					(0,0,0)),
				(x,y))
			y+=14
	
	def center_msg(self, msg):
		for i, line in enumerate(msg.splitlines()):
			msg_image =  self.default_font.render(line, False,
				(255,255,255), (0,0,0))
		
			msgim_center_x, msgim_center_y = msg_image.get_size()
			msgim_center_x //= 2
			msgim_center_y //= 2
		
			self.screen.blit(msg_image, (
			  self.width // 2-msgim_center_x,
			  self.height // 2-msgim_center_y+i*22))
	
	def draw_matrix(self, matrix, offset):
		off_x, off_y  = offset
		for y, row in enumerate(matrix):
			for x, val in enumerate(row):
				if val:
					pygame.draw.rect(
						self.screen,
						colors[val],
						pygame.Rect(
							(off_x+x) *
							  cell_size,
							(off_y+y) *
							  cell_size, 
							cell_size,
							cell_size),0)
	
	def add_cl_lines(self, n):
		linescores = [0, 40, 100, 300, 1200]
		self.lines += n
		self.score += linescores[n] * self.level
		if self.lines >= self.level*6:
			self.level += 1
			newdelay = 1000-50*(self.level-1)
			newdelay = 100 if newdelay < 100 else newdelay
			pygame.time.set_timer(pygame.USEREVENT+1, newdelay)
	
	def move(self, delta_x):
		if not self.gameover and not self.paused:
			new_x = self.stone_x + delta_x
			if new_x < 0:
				new_x = 0
			if new_x > cols - len(self.stone[0]):
				new_x = cols - len(self.stone[0])
			if not check_collision(self.board,
			                       self.stone,
			                       (new_x, self.stone_y)):
				self.stone_x = new_x
	def quit(self):
		self.center_msg("Exiting...")
		pygame.display.update()
		sys.exit()
	
	def drop(self, manual):
		if not self.gameover and not self.paused:
			self.score += 1 if manual else 0
			self.stone_y += 1
			if check_collision(self.board,
			                   self.stone,
			                   (self.stone_x, self.stone_y)):
				self.board = join_matrixes(
				  self.board,
				  self.stone,
				  (self.stone_x, self.stone_y))
				self.new_stone()
				cleared_rows = 0
				while True:
					for i, row in enumerate(self.board[:-1]):
						if 0 not in row:
							self.board = remove_row(
							  self.board, i)
							cleared_rows += 1
							break
					else:
						break
				self.add_cl_lines(cleared_rows)
				return True
		return False
	
	def insta_drop(self):
		if not self.gameover and not self.paused:
			while(not self.drop(True)):
				pass
	
	def rotate_stone(self):
		if not self.gameover and not self.paused:
			new_stone = rotate_clockwise(self.stone)
			if not check_collision(self.board,
			                       new_stone,
			                       (self.stone_x, self.stone_y)):
				self.stone = new_stone
	
	def toggle_pause(self):
		self.paused = not self.paused
	
	def start_game(self):
		if self.gameover:
			self.init_game()
			self.gameover = False


	################# LEARNING AND RUNNING #################################################

	def agentViewMatrix(self):
		result = deepcopy(self.board)

		for y, row in enumerate(self.stone):
			for x, square in enumerate(row):
				result[self.stone_y + y][self.stone_x + x] = square

		return result


	def initializeAgent(self):
		self.session = tf.Session()

		NUM_ACTIONS = 5

		CONV_WIDTH = 4
		CONV_HEIGHT = 9

		FEATURES_LAYER_1 = 64
		FEATURES_LAYER_2 = 128
		NETWORK_OUTPUTS = 512

		LEARNING_RATE = 0.001

		EPOCHS = 100

		GAMA = 0.99




		game_resolution = (cols, rows)
		self.state = tf.placeholder(tf.float32, shape= [None] + list(game_resolution) + [1], name='State')
		self.targets = tf.placeholder(tf.float32, shape=[None, NUM_ACTIONS], name='TargetQ')

		self.conv_width = CONV_WIDTH
		self.conv_height = CONV_HEIGHT
		self.features_layer_1 = FEATURES_LAYER_1
		self.features_layer_2 = FEATURES_LAYER_2
		self.network_outputs = NETWORK_OUTPUTS
		self.learning_rate = LEARNING_RATE
		self.num_actions = NUM_ACTIONS

		self.session = tf.Session()
		self.epochs = EPOCHS
		self.gama = GAMA

	def exploration_rate(self, epoch):
		start_eps = 1.0
		end_eps = 0.1
		const_eps_epochs = 0.1 * self.epochs
		eps_decay_epochs = 0.6 * self.epochs

		if load_model:
			return end_eps
		else:
			if epoch < const_eps_epochs:
				return start_eps
			elif epoch < eps_decay_epochs:
				return start_eps - (epoch - const_eps_epochs) / \
					(eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
			else:
				return end_eps
                
	def perform_learning_step(self, eps, agent_key_actions, get_q_values, learn):
	    
	    s1 = self.agentViewMatrix()

	#    eps = epsilon_list(epoch)
	    if random() <= eps:
	        a = randint(0, len(self.num_actions) - 1)
	    else:
	        a = get_best_action(s1, True)
	        
	    #learning step algorithm
	    self.last_score = self.score
	    agent_key_actions[a]()
	    reward = self.last_score - self.score

	    s2 = self.agentViewMatrix()

	    q2 = np.max(q_values(s2, True), axis=1)
        target_q = q_values(s1, True)

        target_q[np.arange(target_q.shape[0]), a] = reward + self.gama * q2

        learn(s1, target_q, True)
	        
	def sim_perform_step(eps):
	    
	    if random() <= eps:
	        # is_random, random_action, -1 (only to complete number of args)
	        return 1, randint(0, len(actions) - 1), -1
	    else:
	        s1 = preprocess(game.get_state().screen_buffer)
	        # is not random, action without dropout, action with dropout
	        return 0, get_best_action(s1, False), get_best_action(s1, True)

	def create_network():
	    # Add 2 convolutional layers with ReLu activation
	    conv1 = tf.contrib.layers.convolution2d(self.state, num_outputs=self.features_layer_1, kernel_size=[self.conv_width, self.conv_height], stride=[2, 5],
	                                            activation_fn=tf.nn.relu,
	                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
	                                            biases_initializer=tf.constant_initializer(0.1))
	    conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=self.features_layer_2, kernel_size=[self.conv_width, self.conv_height], stride=[2, 5],
	                                            activation_fn=tf.nn.relu,
	                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
	                                            biases_initializer=tf.constant_initializer(0.1))

	    #final layer, between actions and the net
	    conv2_flat = tf.contrib.layers.flatten(conv2)

	    fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=self.network_outputs, activation_fn=tf.nn.relu,
	                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
	                                            biases_initializer=tf.constant_initializer(0.1))

	    is_in_training = tf.placeholder(tf.bool)
	    fc1_drop = tf.contrib.layers.dropout(fc1, keep_prob=0.7, is_training=is_in_training)

	    q = tf.contrib.layers.fully_connected(fc1_drop, num_outputs=self.num_actions, activation_fn=None,
	                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
	                                          biases_initializer=tf.constant_initializer(0.1))
	    best_a = tf.argmax(q, 1)

	    loss = tf.losses.mean_squared_error(q, self.targets)

	    optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
	    # Update the parameters according to the computed gradient using RMSProp.
	    train_step = optimizer.minimize(loss)

	    def function_learn(s1, target_q, is_training):
	        feed_dict = {self.state: s1, self.targets: target_q, is_in_training: is_training}
	        l, _ = self.session.run([loss, train_step], feed_dict=feed_dict)
	        return l

	    def function_get_q_values(state, is_training):
	        return self.session.run(q, feed_dict={self.state: state, is_in_training: is_training})

	    def function_simple_get_q_values(state, is_training):
	        return function_get_q_values(state.reshape([1, cols, rows, 1]), is_training)[0]

	    def function_get_best_action(state, is_training):
	        return self.session.run(best_a, feed_dict={self.state: state, is_in_training: is_training})

	    def function_simple_get_best_action(state, is_training):
	    	return function_get_best_action(state.reshape([1, cols, rows, 1]), is_training)[0]
	    
	    return function_learn, function_get_q_values, function_simple_get_best_action, function_simple_get_q_values

    

	def run(self):
		self.initializeAgent()

		learn, q_values, get_best_action, simple_q = create_network()
		#saver = tf.train.Saver()

    	#key_actions = {
    	#	'ESCAPE':	self.quit,
		#	'LEFT':		lambda:self.move(-1),
		#	'RIGHT':	lambda:self.move(+1),
		#	'DOWN':		lambda:self.drop(True),
		#	'UP':		self.rotate_stone,
		#	'p':		self.toggle_pause,
		#	'SPACE':	self.start_game,
		#	'RETURN':	self.insta_drop
		#}

		agent_key_actions = {
			'LEFT':		lambda:self.move(-1),
			'RIGHT':	lambda:self.move(+1),
			'DOWN':		lambda:self.drop(True),
			'UP':		self.rotate_stone,
			'RETURN':	self.insta_drop
		}
		
		self.gameover = False
		self.paused = False
		
		dont_burn_my_cpu = pygame.time.Clock()

		for epoch in range(self.epochs):
			self.screen.fill((0,0,0))
			if self.gameover:
				if(not train):
					self.center_msg("""Game Over!\nYour score: %dPress space to continue""" % self.score)

				#end of episode here
				else:
					self.start_game()
				####################################################
			else:
				if self.paused:
					self.center_msg("Paused")
				else:
					#pygame.draw.line(self.screen,
						#(255,255,255),
						#(self.rlim+1, 0),
						#(self.rlim+1, self.height-1))
					#self.disp_msg("Next:", (
						#self.rlim+cell_size,
						#2))
					#self.disp_msg("Score: %d\n\nLevel: %d\
#\nLines: %d" % (self.score, self.level, self.lines),
						#(self.rlim+cell_size, cell_size*5))
					#self.draw_matrix(self.bground_grid, (0,0))
					self.draw_matrix(self.board, (0,0))
					self.draw_matrix(self.stone,
						(self.stone_x, self.stone_y))
					self.draw_matrix(self.next_stone,
						(cols+1,0))
			pygame.display.update()


			last_score = self.score

			
			for event in pygame.event.get():
				if event.type == pygame.USEREVENT+1:
					self.drop(False)
				elif event.type == pygame.QUIT:
					self.quit()

			#perform learn step
			if(train):
				eps = self.exploration_rate(epoch, agent_key_actions)
				#self.perform_learning_step(eps, agent_key_actions, get_q_values, learn)

				s1 = self.agentViewMatrix()

			#    eps = epsilon_list(epoch)
				if random() <= eps:
					a = randint(0, len(self.num_actions) - 1)
				else:
					a = get_best_action(s1, True)
			        
			    #learning step algorithm
				agent_key_actions[a]()
				reward = last_score - self.score

				s2 = self.agentViewMatrix()

				q2 = np.max(q_values(s2, True), axis=1)
				target_q = q_values(s1, True)

				target_q[np.arange(target_q.shape[0]), a] = reward + self.gama * q2

				learn(s1, target_q, True)


				#elif event.type == pygame.KEYDOWN and play_game:
				#	for key in key_actions:
				#		if event.key == eval("pygame.K_"+key):
				#			key_actions[key]()

			#if(not play_game):
			#	for key in agent_key_actions:
			#		if(randint(0,99) < 15):
			#			agent_key_actions[key]()

			#for row in self.agentViewMatrix():
			#	print (row)

			#clear()
					
			dont_burn_my_cpu.tick(maxfps)

if __name__ == '__main__':
	App = TetrisApp()
	App.run()

