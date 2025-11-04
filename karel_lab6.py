# karel.py
import time
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import simpleaudio as sa
import pygame

class KarelPupper:
    def start():
        if not rclpy.ok():
            rclpy.init()

    def __init__(self):
        if not rclpy.ok():
            rclpy.init()
        self.node = Node('karel_node')
        self.publisher = self.node.create_publisher(Twist, 'cmd_vel', 10)

    def move(self, linear_x, linear_y, angular_z):
        move_cmd = Twist()
        move_cmd.linear.x = linear_x
        move_cmd.linear.y = linear_y
        move_cmd.angular.z = angular_z
        self.publisher.publish(move_cmd)
        rclpy.spin_once(self.node, timeout_sec=1.0)
        self.stop()
    
    def wiggle(self, wiggle_time=6, play_sound=True):
        # Play wiggle sound if requested
        if play_sound:
            pygame.mixer.init()
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sounds_dir = os.path.join(current_dir, '..', '..', 'sounds')
            wav_path = os.path.join(sounds_dir, 'puppy_wiggle.wav')
            wav_path = os.path.normpath(wav_path)
            
            try:
                wiggle_sound = pygame.mixer.Sound(wav_path)
                wiggle_sound.play()
                self.node.get_logger().info(f'Playing wiggle sound from: {wav_path}')
            except Exception as e:
                self.node.get_logger().warning(f"Could not play wiggle sound: {e}")

        move_cmd = Twist()
        move_cmd.linear.x = 0.0
        # Alternate wiggle directions for a total of 1 second
        single_wiggle_duration = 0.2  # seconds per half-wiggle
        angular_speed = 0.8
        
        start_time = time.time()
        direction = 1
        while time.time() - start_time < wiggle_time:
            move_cmd.angular.z = direction * angular_speed
            self.publisher.publish(move_cmd)
            rclpy.spin_once(self.node, timeout_sec=0.01)
            time.sleep(single_wiggle_duration)
            direction *= -1  # Switch direction
        
        self.stop()

        self.node.get_logger().info('Wiggle!')
    
    def bob(self, bob_time=5, play_sound=True):
        """
        Makes the robot bob back and forth by moving forward and backward with a specified speed and duration.

        TODO: 
        1. Play a 'puppy_bob.wav' sound if play_sound is True.
            - Use pygame.mixer to initialize the sound engine.
            - Load the 'puppy_bob.wav' file from the sounds directory.
            - Play the sound and handle any exceptions gracefully, logging them with self.node.get_logger().
        2. Publish alternating Twist messages to make the robot bob forward and backward.
            - Bob back and forth with a configurable speed and duration (bob_time).
            - Alternate the direction of linear.x every 0.2 seconds (half_bob_duration).
            - Call rclpy.spin_once and use time.sleep to manage timing.
        3. Call self.stop() at the end to halt the robot.

        Remove the 'pass' statement after you implement the steps above.
        """
        # ==== TODO: Implement the steps above ====
        if play_sound:
            pygame.mixer.init()
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sounds_dir = os.path.join(current_dir, '..', '..', 'sounds')
            wav_path = os.path.join(sounds_dir, 'puppy_bob.wav')
            wav_path = os.path.normpath(wav_path)

            try:
                bob_sound = pygame.mixer.Sound(wav_path)
                bob_sound.play()
                self.node.get_logger().info(f'Playing bob sound from: {wav_path}')
            except Exception as e:
                self.node.get_logger().warning(f"Could not play bob sound: {e}")

        move_cmd = Twist()
        move_cmd.angular.z = 0.0
        single_bob_duration = 0.3

        start_time = time.time()
        direction = 1
        while time.time() - start_time < bob_time:
            move_cmd.linear.x = direction * 0.2
            move_cmd.linear.y = -direction * 0.1
            self.publisher.publish(move_cmd)
            time.sleep(single_bob_duration)
            direction *= -1

        self.stop()

        self.node.get_logger().info('Bob!')


        self.node.get_logger().info('Bob!')

    def move_forward(self):
        """
        TODO: Implement moving Pupper forward.
        - Decide on an appropriate linear.x speed for safe forward movement.
        - Use the move() helper function that is implemented above, or manually construct move_cmd = Twist().
        - Publish the Twist command for a set duration, then stop.
        """
        move_cmd = Twist()
        move_cmd.linear.x = 1.0
        move_cmd.angular.z = 0.0 
        self.publisher.publish(move_cmd)
        rclpy.spin_once(self.node, timeout_sec=1.0)
        self.node.get_logger().info('Move forward...')
        self.stop()

    def move_backward(self):
        """
        TODO: Implement moving Pupper backward.
        - Decide on a negative linear.x value for safe backward movement.
        - Use move() or create your own Twist message.
        - Be careful with speed—backward motion is often best slower.
        """
        move_cmd = Twist()
        move_cmd.linear.x = -1.0
        move_cmd.angular.z = 0.0 
        self.publisher.publish(move_cmd)
        rclpy.spin_once(self.node, timeout_sec=1.0)
        self.node.get_logger().info('Move backward...')
        self.stop()

    def move_left(self):
        """
        TODO: Implement moving Pupper to the left (translation).
        - Set an appropriate linear.y value for left strafe.
        - Use move() or build the move_cmd yourself.
        """
        self.turn_left()
        self.move_forward()
        self.stop()


    def move_right(self):
        """
        TODO: Implement moving Pupper to the right (translation).
        - Set an appropriate negative linear.y value for right strafe.
        - Use move() or create your own move_cmd.
        """
        self.turn_right()
        self.move_forward()
        self.stop()

    def turn_left(self):
        """
        TODO: Implement turning Pupper left (rotation).
        - Set a positive angular.z value for left rotation.
        - Use move() or build your own move_cmd.
        """
        move_cmd = Twist()
        move_cmd.angular.z = 1.5
        self.publisher.publish(move_cmd)
        rclpy.spin_once(self.node, timeout_sec=1)
        self.stop()

    def turn_right(self):
        """
        TODO: Implement turning Pupper right (rotation).
        - Set a negative angular.z value for right rotation.
        - Use move() or make your own Twist message.
        """
        move_cmd = Twist()
        move_cmd.angular.z = -1.5
        self.publisher.publish(move_cmd)
        rclpy.spin_once(self.node, timeout_sec=0.01)  
        self.stop()


    def bark(self):
        self.node.get_logger().info('Bark...')
        pygame.mixer.init()
        # Directory-independent path to sound file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sounds_dir = os.path.join(current_dir, '..', '..', 'sounds')
        bark_sound_path = os.path.join(sounds_dir, 'dog_bark.wav')

        bark_sound_path = os.path.normpath(bark_sound_path)
        bark_sound = pygame.mixer.Sound(bark_sound_path)
        bark_sound.play()
        self.node.get_logger().info(f'Playing bark sound from: {bark_sound_path}')
        self.stop()
    
    def dance(self):
        self.node.get_logger().info('Rick Rolling...')
        pygame.mixer.init()
        # Directory-independent path to sound file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sounds_dir = os.path.join(current_dir, '..', '..', 'sounds')
        dance_sound_path = os.path.join(sounds_dir, 'rickroll.wav')

        dance_sound_path = os.path.normpath(dance_sound_path)
        dance_sound = pygame.mixer.Sound(dance_sound_path)
        self.node.get_logger().info(f'Playing dance sound from: {dance_sound_path}')
        dance_sound.play()
        # TODO: Create your own awesome Pupper dance move sequence here!
        # Use combinations of self.wiggle(), self.turn_left(), self.turn_right(), self.bob(), and self.stop().
        # Be creative and choreograph the most exciting dance possible!
        pass


    def stop(self):
        self.node.get_logger().info('Stopping...')
        move_cmd = Twist()
        move_cmd.linear.x = 0.0
        move_cmd.linear.y = 0.0
        move_cmd.linear.z = 0.0
        move_cmd.angular.x = 0.0
        move_cmd.angular.y = 0.0
        move_cmd.angular.z = 0.0
        self.publisher.publish(move_cmd)
        rclpy.spin_once(self.node, timeout_sec=1.0)
    
    def __del__(self):
        self.node.get_logger().info('Tearing down...')
        self.node.destroy_node()
        rclpy.shutdown()