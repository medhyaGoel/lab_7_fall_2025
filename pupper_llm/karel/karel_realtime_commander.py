#!/usr/bin/env python3
"""
Karel Realtime Commander
Simplified commander for use with OpenAI Realtime API.
Focuses on command extraction and robot control since voice/LLM is handled by Realtime API.
"""

import asyncio
import re
import logging
from typing import Optional, Tuple
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String
import karel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("karel_realtime_commander")


class KarelRealtimeCommanderNode(Node):
    """Simplified commander that extracts and executes commands from Realtime API responses."""
    
    def __init__(self):
        super().__init__('karel_realtime_commander_node')
        
        # Subscribe to responses from Realtime API
        self.response_subscription = self.create_subscription(
            String,
            'gpt4_response_topic',
            self.response_callback,
            10
        )
        
        # Also subscribe to transcription for logging
        self.transcription_subscription = self.create_subscription(
            String,
            '/transcription',
            self.transcription_callback,
            10
        )
        
        # Initialize robot
        self.pupper = karel.KarelPupper()
        
        # Command queue with timestamps
        self.command_queue = asyncio.Queue()
        self.processing_commands = False
        self.command_timeout = 20.0  # Clear commands older than 20 seconds
        
        logger.info('Karel Realtime Commander initialized')
        logger.info('Listening for commands from Realtime API...')
    
    def transcription_callback(self, msg):
        """Log user transcriptions."""
        logger.debug(f"👤 User: {msg.data}")
    
    def response_callback(self, msg):
        """Process responses and extract commands line by line."""
        response = msg.data
        logger.info(f"🤖 Response: {response}")
        all_commands = []
        
        # TODO: Paste your Lab 6 command parsing implementation here
        # Parse commands from the response text line by line and dispatch them in order into the `all_commands` list.
        # 1. Split the `response` string into lines using `\n` as a separator.
        # 2. For each line that is not blank, call `self.extract_commands_from_line(line.strip())` to get a list of commands from that line.
        # 3. Collect all commands, preserving the original order.
        # 4. Append each command to `all_commands` (should be a flat list, not nested).
        # 5. This ensures that multi-line responses generate a sequence of actions in the same order as the LLM output.
        # Example:
        #   If response is:
        #     "Move forward\nTurn left\nBark"
        #   Your code should process:
        #     ["move", "turn_left", "bark"]

        # Your code here:
        lines = response.split('\n')
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:  # Ensure line is not blank
                commands_from_line = self.extract_commands_from_line(stripped_line)
                # Use extend to add elements from the returned list, keeping the list flat
                all_commands.extend(commands_from_line)

        
        if all_commands:
            logger.info(f"📋 Commands (in order): {all_commands}")
            # Queue commands with timestamp in sequential order
            current_time = time.time()
            for cmd in all_commands:
                command_with_time = (cmd, current_time)
                asyncio.create_task(self.command_queue.put(command_with_time))
        else:
            logger.debug("No commands found")
    
    def extract_commands_from_line(self, line: str) -> list:
        """
        TODO: Paste your Lab 6 implementation and extend it for tracking commands.

        HINTS:
        - The parsing logic will depend on exactly how you format your GPT model/system prompt!
        - The commands you define here will be used to execute the commands in the execute_command function, which you'll also implement below.
        - Think carefully about the sequential structure of the LLM's output and how your system prompt tells the model to format commands.
        - For example, if your prompt instructs GPT to output one command per line, you should parse only a single command from each line.
        - You may need to match/substitute multiple possible phrasings (e.g. "move forward", "walk forward") to a canonical command like "move".
        
        NEW FOR LAB 7 - Tracking Commands:
        - Add detection for "start tracking [object]" or "track the [object]" or "follow the [object]"
        - Extract the object name (e.g., "person", "dog", "cat") from the phrase
        - Return a command like "track_person" or "track_dog" (format: "track_{object_name}")
        - Add detection for "stop tracking" or "stop following" → return "stop_tracking"
        
        Example tracking command parsing:
            line = "Start tracking person"
            → extract "person" and return ["track_person"]
            
            line = "Follow the dog"
            → extract "dog" and return ["track_dog"]
            
            line = "Stop tracking"
            → return ["stop_tracking"]
        
        - Construct and return a list of action strings (e.g. ['move', 'turn_left']) extracted from the line.
        - Test your command extraction logic carefully, since if the output is not sequential, your robot may behave out of order!

        Example:
            line = "Move forward"
            returns ['move']

            line = "<move, turn_left>"
            returns ['move', 'turn_left']
        """
        VALID_COMMANDS = {"move", "left", "right", "dance", "wiggle", "bark", "stop", "stop_tracking_{object}", "track_{object}"}
        
        cleaned_line = line.strip().lower()
        
        # Because the system_prompt is so strict, we only need to check
        # if the cleaned line is one of the valid action keywords.
        if cleaned_line in VALID_COMMANDS:
            return [cleaned_line]
        
        # If the line is "None", blank, or any other text, it's not a valid command.
        return []
    
    async def execute_command(self, command: str) -> bool:
        """Execute a single robot command."""
        try:
            logger.info(f"⚙️  Executing {command}")
            
            # NEW FOR LAB 7: Handle tracking commands
            # TODO: Add tracking command handling BEFORE your Lab 6 command mappings
            # - If command starts with "track_", extract the object name and call self.pupper.begin_tracking(object_name)
            #   Example: if command is "track_person", extract "person" and call self.pupper.begin_tracking("person")
            #   Use: object_name = command.split("_", 1)[1]
            # - If command is "stop_tracking", call self.pupper.end_tracking()
            # - Use await asyncio.sleep(0.5) after each tracking command
            if command.startswith("track_"):
                object_name = command.split("_", 1)[1]
                self.pupper.begin_tracking(object_name)
                await asyncio.sleep(0.5)
            elif command.startswith("stop_tracking_"):
                object_name = command.split("_", 1)[1]
                self.pupper.end_tracking(object_name)
                await asyncio.sleep(0.5)
            
            # TODO: Paste your Lab 6 command mapping implementation below
            # Implement the mapping from canonical command names (e.g., "move", "turn_left", "bark", etc.) to the appropriate KarelPupper action and its timing.
            # One complete mapping is shown as an example!
            if command in ["move", "go", "forward"]:
                self.pupper.move_forward()
                await asyncio.sleep(0.5)  # Hint: Use await asyncio.sleep(seconds) to pace each action!
            # TODO: Add additional elifs for the other actions that KarelPupper supports,
            #       calling the correct pupper method, and using an appropriate sleep time after each command.
            # For example:
            #   - For "wiggle"/"wag" actions, the total animation can take ~5.5 seconds; use await asyncio.sleep(5.5)
            #   - For "bob" actions, the action can take ~5.5 seconds; use await asyncio.sleep(5.5)
            #   - For "dance" actions, the full dance is ~12.0 seconds; use await asyncio.sleep(12.0)
            #   - For most normal moves and turns, use 0.5 seconds.
            # See the KarelPupper API for supported commands and their method names.
            elif command in ["left", "turn_left"]:
                self.pupper.turn_left()
                await asyncio.sleep(0.5)
            elif command in ["right", "turn_right"]:
                self.pupper.turn_right()
                await asyncio.sleep(0.5)
            elif command in ["dance"]:
                self.pupper.dance()
                await asyncio.sleep(12.0)
            elif command in ["wiggle", "wag"]:
                self.pupper.wiggle()
                await asyncio.sleep(5.5)
            elif command in ["bark", "speak"]:
                self.pupper.bark()
                await asyncio.sleep(0.5)
            elif command in ["stop", "halt"]:
                self.pupper.stop()
                await asyncio.sleep(0.1)
            else:
                logger.warning(f"⚠️  Unknown command: {command}")
                return False
            
            logger.info(f"✅ Done")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return False
    
    async def command_processor_loop(self):
        """Process commands from the queue with timeout checking."""
        logger.info("🔄 Command processor started")
        
        while rclpy.ok():
            try:
                # Get next command with timestamp (wait up to 0.1s)
                command_data = await asyncio.wait_for(
                    self.command_queue.get(),
                    timeout=0.1
                )
                
                # Unpack command and timestamp
                command, timestamp = command_data
                
                # Check if command is stale (older than 20 seconds)
                age = time.time() - timestamp
                if age > self.command_timeout:
                    logger.warning(f"⏰ Discarding stale command '{command}' (age: {age:.1f}s)")
                    continue
                
                # Execute command
                await self.execute_command(command)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in command processor: {e}")
                await asyncio.sleep(0.1)
    
    async def run(self):
        """Main run loop."""
        await self.command_processor_loop()


async def main_async(args=None):
    """Async main function."""
    rclpy.init(args=args)
    
    node = KarelRealtimeCommanderNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    
    try:
        logger.info("🚀 Karel Realtime Commander started")
        logger.info("Ready to receive commands from Realtime API")
        
        # Create tasks
        ros_task = asyncio.create_task(spin_ros_async(executor))
        command_task = asyncio.create_task(node.run())
        
        await asyncio.gather(ros_task, command_task)
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        executor.shutdown()
        rclpy.shutdown()


async def spin_ros_async(executor):
    """Spin ROS2 executor in async-friendly way."""
    while rclpy.ok():
        executor.spin_once(timeout_sec=0.1)
        await asyncio.sleep(0.01)


def main(args=None):
    """Entry point."""
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("Program interrupted")


if __name__ == '__main__':
    main()

