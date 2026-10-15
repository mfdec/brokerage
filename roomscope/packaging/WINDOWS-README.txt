RoomScope - room acoustics analyzer for Windows 10/11 (64-bit)

START
  1. Unzip the whole folder somewhere (e.g. C:\RoomScope). Do not move RoomScope.exe out of the folder:
     it needs the "_internal" folder next to it.
  2. Double-click RoomScope.exe.
     Windows may say "Windows protected your PC" because the program is not code-signed:
     click "More info", then "Run anyway".
  3. Try it without hardware first: in the top bar choose "Simulated room (no hardware)" and press Start listening.
     On the Camera map tab choose "Simulated camera (demo)", press Connect, then "Demo: place points automatically",
     set the lens FOV to 70, and use the "Demo: virtual walk" button further down.

FIRST-TIME SETUP (real hardware)
  * Plug in the Dayton iMM-6C. In Windows Settings > System > Sound > Input > (the mic) > Properties:
    turn "Audio enhancements" OFF and leave the input level at a fixed value.
  * Windows Settings > Privacy & security > Microphone: allow desktop apps to use the microphone.
  * In RoomScope: pick the mic in the top bar, load your mic calibration file ("Mic cal file"), press Start listening.
  * On the Measure / Decay tab choose your speakers as the output device.

IF SOMETHING DOES NOT WORK
  Open a Command Prompt or PowerShell in this folder and run:
      RoomScope.exe --doctor --all        (checks audio devices, the mic's noise floor and cameras)
      RoomScope.exe --selftest            (checks the program itself with simulated hardware, ~20 s)
  and send the output.

This version (0.6) has been tested against a simulated room, camera and walker only; it has not yet been run with a
real microphone or camera. See docs\VERIFICATION.md in the source repository for the checklist.
