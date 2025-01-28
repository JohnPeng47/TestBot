import os
import sys
import termios
import tty
from typing import Optional, Any, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

class IOColor(Enum):
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

class IO:
    """Handles terminal I/O operations with proper TTY management"""
    
    def __init__(self, tty_path: str = '/dev/tty'):
        self.tty_path = self._get_tty_path(tty_path)
        self.tty_file = None
        self.original_settings = None
        self.original_stdin_fd = None
        
    def _get_tty_path(self, preferred_path: str) -> str:
        """Find an available TTY device"""
        tty_paths = [preferred_path, '/dev/tty', '/dev/tty0', '/dev/tty1']
        for path in tty_paths:
            if os.path.exists(path):
                return path
        raise RuntimeError("No TTY device available")

    def _setup_tty(self):
        """Set up TTY for input"""
        # Open TTY by file descriptor instead of file object
        self.tty_fd = os.open(self.tty_path, os.O_RDWR)
        try:
            # Get and save original terminal settings
            self.original_settings = termios.tcgetattr(self.tty_fd)
            
            # Modify settings for raw mode
            new_settings = termios.tcgetattr(self.tty_fd)
            new_settings[3] = new_settings[3] & ~termios.ECHO & ~termios.ICANON  # lflags
            termios.tcsetattr(self.tty_fd, termios.TCSANOW, new_settings)
            
        except Exception as e:
            if self.tty_fd is not None:
                os.close(self.tty_fd)
                self.tty_fd = None
            raise RuntimeError(f"Failed to setup TTY: {e}")

    def _cleanup_tty(self):
        """Clean up TTY settings"""
        if self.tty_fd is not None:
            try:
                if self.original_settings is not None:
                    termios.tcsetattr(self.tty_fd, termios.TCSADRAIN, self.original_settings)
            finally:
                os.close(self.tty_fd)
                self.tty_fd = None
                self.original_settings = None

    def _colorize(self, text: str, color: IOColor) -> str:
        """Add color to text"""
        return f"{color.value}{text}{IOColor.RESET.value}"

    def _write(self, text: str, color: IOColor = IOColor.RESET):
        """Write to TTY"""
        text = self._colorize(text, color)
        if self.tty_fd is not None:
            os.write(self.tty_fd, text.encode())

    def _read_char(self) -> str:
        """Read a single character from TTY"""
        if self.tty_fd is not None:
            return os.read(self.tty_fd, 1).decode()
        return ''

    def _read_line(self) -> str:
        """Read a line from TTY"""
        if self.tty_fd is None:
            return ''
            
        response = []
        while True:
            char = self._read_char()
            if char in ('\n', '\r'):
                self._write('\n')  # Echo newline
                break
            if char == '\x03':  # Ctrl-C
                raise KeyboardInterrupt
            if char == '\x04':  # Ctrl-D
                if not response:
                    raise EOFError
            if char == '\x7f':  # Backspace
                if response:
                    response.pop()
                    self._write('\b \b')  # Erase character
                continue
            if char.isprintable():
                response.append(char)
                self._write(char)  # Echo character
                
        return ''.join(response).strip()

    def input(self, 
              prompt: str, 
              validator: Optional[Callable[[str], Any]] = lambda x: x) -> str:
        """Get user input with optional validation"""
        try:
            self._setup_tty()
            self._write(prompt, color=IOColor.YELLOW)
            
            response = self._read_line()
            return validator(response)
            
        except Exception as e:
            import traceback
            raise RuntimeError(f"Failed to get user input: {e}\nStacktrace:\n{traceback.format_exc()}")
        finally:
            self._cleanup_tty()

    def confirm_or_exit(self, message: str) -> None:
        """Prompt for confirmation or exit the program"""
        if not self.prompt(message):
            sys.exit(1)

    def __del__(self):
        """Ensure cleanup on object destruction"""
        self._cleanup_tty()
