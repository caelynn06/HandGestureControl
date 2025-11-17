# actions.py
"""
System actions for gesture control
"""
import subprocess # for running system commands
import platform # for detecting operating system
import os # for operating system interaction
import pyautogui # for keyboard control


def open_myutep(): 
    """Open my.utep.edu in default browser"""
    url = "https://my.utep.edu" # feel free to change the link!
    system = platform.system()  # Detect which operating system we're running on

    try:
        if system == "Windows":
            subprocess.Popen(['start', url], shell=True)
        elif system == "Darwin": # macOS
            subprocess.Popen(['open', url])
        else:
            subprocess.Popen(['xdg-open', url])
        print("Opened my UTEP")
    except Exception as e:
        print("Error opening UTEP:", e)


def open_spotify(): # os system calls spotify
    """Open Spotify application"""
    print("Opening Spotify")
    try:
        os.system("start spotify") # feel free to customize!
    except Exception as e:
        print("Error opening Spotify:", e)


def change_volume(direction):
    """
    Change system volume
    Args:
        direction: "UP" or "DOWN"
    """
    system = platform.system()
    try:
        if system in ("Windows", "Darwin"):
            pyautogui.press("volumeup" if direction == "UP" else "volumedown")
        else:
            step = "5%+" if direction == "UP" else "5%-" # increase/decrease volume by 5%
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", step])
            # "@DEFAULT_SINK@" means the default audio output device

    except Exception as e:
        print("Volume error:", e)

