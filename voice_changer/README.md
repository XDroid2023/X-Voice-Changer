# X Voice Changer

A professional-grade voice manipulation application with 24 unique voice effects and real-time audio processing.

## Features

- Real-time voice recording with status indicator
- Modern dark theme UI with Twitter blue accents
- 24 unique voice effects including:
  - Base Effects: Normal, Alien, Robot, Chipmunk, Demon, Underwater, Radio, Chorus
  - Digital Effects: Cyberpunk, Cave, 8-Bit, Telephone
  - Special Effects: Stadium, Vinyl, Vocoder, Space
  - Fantasy Effects: Dragon, Crystal, Storm
  - Experimental: Time-Stretch, Reverse-Echo, Metallic, Ghostly, Quantum
- Advanced parameter controls:
  - Pitch (0.5x to 2.0x)
  - Reverb
  - Distortion
  - Delay
  - Flanger

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python voice_changer.py
   ```

## Usage

1. Click "START" to begin recording
2. Speak into your microphone
3. Click "STOP" to end recording
4. Select a voice effect
5. Adjust parameters with sliders
6. Click "PLAY" to hear your voice

## Requirements

- Python 3.7+
- Dependencies:
  - numpy>=1.24.3
  - scipy>=1.10.1
  - sounddevice>=0.4.6
- Working microphone
- Audio output device

## Privacy & Security

- All audio processing is done locally
- No data is sent to external servers
- No recordings are saved permanently

## License

MIT License - feel free to use and modify for your own projects!
